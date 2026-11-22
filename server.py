import json
import math
import os
import re
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from dense_retrieval import DenseRetriever
from evaluation_analysis import build_evaluation_analysis


ROOT = Path(__file__).resolve().parent
FRONT_DIR = ROOT / "front"

app = Flask(__name__)


def env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return default
    return value not in {"0", "false", "no", "off"}


def format_count(value: int) -> str:
    return f"{value:,}".replace(",", " ")


@app.after_request
def disable_front_cache(response):
    if request.path.startswith("/api/"):
        return response
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


TOKEN_PATTERN = re.compile(r"[a-z]+|[0-9]+(?:\.[0-9]+)?")
FALLBACK_STOPWORDS = {
    "a", "about", "an", "and", "are", "as", "at", "be", "but", "by",
    "for", "from", "has", "have", "how", "i", "in", "is", "it", "not",
    "of", "on", "or", "that", "the", "this", "to", "was", "what", "when",
    "where", "which", "with", "you",
}

try:
    import nltk
    from nltk.corpus import stopwords

    try:
        STOPWORDS = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        STOPWORDS = set(stopwords.words("english"))
except (ImportError, LookupError):
    STOPWORDS = FALLBACK_STOPWORDS


class SearchEngine:
    def __init__(self) -> None:
        self.documents: list[dict] = []
        self.jaccard_index: dict[str, set[str]] = defaultdict(set)
        self.document_sizes: Counter = Counter()
        self.tfidf_index: dict[str, dict[str, float]] = defaultdict(dict)
        self.tfidf_idf: dict[str, float] = {}
        self.tfidf_norms: dict[str, float] = {}
        self.bm25_frequencies: dict[str, dict[str, int]] = defaultdict(dict)
        self.bm25_idf: dict[str, float] = {}
        self.document_lengths: dict[str, int] = {}
        self.average_document_length = 0.0
        self.ready = False
        self.loading = False
        self.error = ""
        self.source = ""
        self.lock = threading.RLock()
        self.relevant_ids: set[str] = set()
        self.evaluation_ready = False
        self.evaluation_loading = False
        self.evaluation_error = ""
        self.evaluation_summary: dict = {}
        self.evaluation_groups: dict = {}
        self.evaluation_queries: dict = {}
        self.evaluation_classified: dict = {}
        self.evaluation_profiles: dict = {}
        self.evaluation_generation = 0
        self.dense_evaluation_loading = False
        self.dense_evaluation_error = ""
        self.dense_evaluation_profiles: dict = {}
        self.dense = DenseRetriever(
            build_if_missing=env_flag("DENSE_BUILD_IF_MISSING", True),
        )

    def tokenize(self, text: str) -> list[str]:
        return [
            token
            for token in TOKEN_PATTERN.findall(str(text).lower())
            if token not in STOPWORDS
        ]

    def import_documents(self, documents: list[dict], source: str) -> None:
        with self.lock:
            if self.loading:
                raise RuntimeError("La importacion ya esta en progreso")
            self.evaluation_generation += 1
            self.evaluation_ready = False
            self.evaluation_loading = False
            self.evaluation_error = ""
            self.evaluation_summary = {}
            self.evaluation_groups = {}
            self.evaluation_queries = {}
            self.evaluation_classified = {}
            self.evaluation_profiles = {}
            self.dense_evaluation_loading = False
            self.dense_evaluation_error = ""
            self.dense_evaluation_profiles = {}
            self.dense.reset()
            self.loading = True
            self.error = ""
            self.ready = False
            self.source = source

        try:
            cleaned = []
            for position, document in enumerate(documents, start=1):
                doc_id = str(document.get("_id") or document.get("id") or position)
                title = str(document.get("title") or "")
                text = str(document.get("text") or document.get("content") or "")
                cleaned.append({"_id": doc_id, "title": title, "text": text})

            jaccard_index = defaultdict(set)
            document_sizes = Counter()
            tfidf_frequencies = defaultdict(dict)
            bm25_frequencies = defaultdict(dict)
            document_lengths = {}

            for document in cleaned:
                doc_id = document["_id"]
                text_tokens = self.tokenize(document["text"])
                for term in set(text_tokens):
                    jaccard_index[term].add(doc_id)
                document_sizes[doc_id] = len(set(text_tokens))
                for term, frequency in Counter(text_tokens).items():
                    tfidf_frequencies[term][doc_id] = frequency

            bm25_tokens_by_document = {
                document["_id"]: self.tokenize(document["text"])
                for document in cleaned
            }
            bm25_frequencies = defaultdict(dict)
            document_lengths = {}
            for doc_id, tokens in bm25_tokens_by_document.items():
                document_lengths[doc_id] = len(tokens)
                for term, frequency in Counter(tokens).items():
                    bm25_frequencies[term][doc_id] = frequency

            document_count = len(cleaned)
            bm25_idf = {}
            total_length = sum(document_lengths.values())
            average_document_length = (
                total_length / document_count if document_count else 0.0
            )
            for term, documents_with_term in bm25_frequencies.items():
                df = len(documents_with_term)
                bm25_idf[term] = math.log1p(
                    (document_count - df + 0.5) / (df + 0.5)
                )

            tfidf_index = defaultdict(dict)
            tfidf_idf = {}
            tfidf_norms = defaultdict(float)
            for term, document_frequencies in tfidf_frequencies.items():
                tfidf_idf[term] = math.log(
                    (document_count + 1) / (len(document_frequencies) + 1)
                ) + 1
                for doc_id, frequency in document_frequencies.items():
                    weight = frequency * tfidf_idf[term]
                    tfidf_index[term][doc_id] = weight
                    tfidf_norms[doc_id] += weight * weight

            tfidf_norms = {
                doc_id: math.sqrt(total)
                for doc_id, total in tfidf_norms.items()
                if total
            }

            with self.lock:
                self.documents = cleaned
                self.jaccard_index = jaccard_index
                self.document_sizes = document_sizes
                self.tfidf_index = tfidf_index
                self.tfidf_idf = tfidf_idf
                self.tfidf_norms = tfidf_norms
                self.bm25_frequencies = bm25_frequencies
                self.bm25_idf = bm25_idf
                self.document_lengths = document_lengths
                self.average_document_length = (
                    average_document_length
                )
                self.relevant_ids = set()
                self.ready = True
        except Exception as exc:
            with self.lock:
                self.error = str(exc)
                self.ready = False
        finally:
            with self.lock:
                self.loading = False

    def import_huggingface(self, dataset_name: str = "BeIR/fiqa", split: str = "corpus") -> None:
        try:
            from datasets import load_dataset

            corpus = load_dataset(dataset_name, "corpus", split=split)
            documents = [dict(document) for document in corpus]
            self.import_documents(documents, f"Hugging Face: {dataset_name}/{split}")
            if self.ready:
                self.load_fiqa_evaluation()
        except Exception as exc:
            with self.lock:
                self.loading = False
                self.ready = False
                self.error = str(exc)

    def get_document(self, doc_id: str) -> dict | None:
        for document in self.documents:
            if document["_id"] == doc_id:
                return document
        return None

    def search_jaccard(self, query: str, k: int) -> list[dict]:
        query_terms = set(self.tokenize(query))
        intersections = Counter()
        for term in query_terms:
            intersections.update(self.jaccard_index.get(term, ()))

        results = []
        for doc_id, intersection in intersections.items():
            document_size = self.document_sizes[doc_id]
            union = document_size + len(query_terms) - intersection
            score = intersection / union if union else 0.0
            results.append((doc_id, score))

        results.sort(key=lambda item: (-item[1], item[0]))
        return self.format_hits(results[:k], query)

    def search_tfidf(self, query: str, k: int) -> list[dict]:
        query_weights = {
            term: frequency * self.tfidf_idf[term]
            for term, frequency in Counter(self.tokenize(query)).items()
            if term in self.tfidf_idf
        }
        if not query_weights:
            return []

        query_norm = math.sqrt(sum(weight * weight for weight in query_weights.values()))
        products = defaultdict(float)
        for term, query_weight in query_weights.items():
            for doc_id, document_weight in self.tfidf_index[term].items():
                products[doc_id] += query_weight * document_weight

        results = []
        for doc_id, product in products.items():
            norm = self.tfidf_norms.get(doc_id)
            if norm:
                results.append((doc_id, product / (query_norm * norm)))

        results.sort(key=lambda item: (-item[1], item[0]))
        return self.format_hits(results[:k], query)

    def search_bm25(self, query: str, k: int, k1: float = 1.2, b: float = 0.75) -> list[dict]:
        scores = defaultdict(float)
        query_terms = dict.fromkeys(self.tokenize(query))
        for term in query_terms:
            document_frequencies = self.bm25_frequencies.get(term, {})
            if not document_frequencies:
                continue
            idf = self.bm25_idf[term]
            for doc_id, frequency in document_frequencies.items():
                length = self.document_lengths.get(doc_id, 0)
                normalization = k1 * (
                    1 - b + b * length / self.average_document_length
                    if self.average_document_length
                    else k1
                )
                scores[doc_id] += idf * (
                    frequency * (k1 + 1) / (frequency + normalization)
                )

        results = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return self.format_hits(results[:k], query)

    def search_dense(self, query: str, k: int) -> list[dict]:
        return self.format_hits(self.dense.search(query, k), query)

    def dense_ready(self) -> bool:
        return self.dense.ensure_ready(self.documents)

    def dense_status(self) -> dict:
        status = self.dense.status()
        state = status["state"]
        if state == "ready":
            message = ""
        elif state == "building":
            message = (
                f"Generando embeddings… "
                f"{format_count(status['done'])}/{format_count(status['total'])}"
            )
        elif state == "error":
            message = f"Error en el índice semántico: {status['error']}"
        else:
            message = "Cargando índice semántico…"
        return {**status, "message": message}

    def ranking_jaccard(self, query: str, k: int) -> list[tuple[str, float]]:
        query_terms = set(self.tokenize(query))
        intersections = Counter()
        for term in query_terms:
            intersections.update(self.jaccard_index.get(term, ()))
        results = []
        for doc_id, intersection in intersections.items():
            document_size = self.document_sizes[doc_id]
            union = document_size + len(query_terms) - intersection
            score = intersection / union if union else 0.0
            results.append((doc_id, score))
        results.sort(key=lambda item: (-item[1], item[0]))
        return results[:k]

    def ranking_tfidf(self, query: str, k: int) -> list[tuple[str, float]]:
        query_weights = {
            term: frequency * self.tfidf_idf[term]
            for term, frequency in Counter(self.tokenize(query)).items()
            if term in self.tfidf_idf
        }
        if not query_weights:
            return []
        query_norm = math.sqrt(sum(weight * weight for weight in query_weights.values()))
        products = defaultdict(float)
        for term, query_weight in query_weights.items():
            for doc_id, document_weight in self.tfidf_index[term].items():
                products[doc_id] += query_weight * document_weight
        results = []
        for doc_id, product in products.items():
            norm = self.tfidf_norms.get(doc_id)
            if norm:
                results.append((doc_id, product / (query_norm * norm)))
        results.sort(key=lambda item: (-item[1], item[0]))
        return results[:k]

    def ranking_bm25(self, query: str, k: int, k1: float = 1.2, b: float = 0.75) -> list[tuple[str, float]]:
        scores = defaultdict(float)
        for term in dict.fromkeys(self.tokenize(query)):
            document_frequencies = self.bm25_frequencies.get(term, {})
            if not document_frequencies:
                continue
            idf = self.bm25_idf[term]
            for doc_id, frequency in document_frequencies.items():
                length = self.document_lengths.get(doc_id, 0)
                normalization = k1 * (
                    1 - b + b * length / self.average_document_length
                    if self.average_document_length
                    else k1
                )
                scores[doc_id] += idf * frequency * (k1 + 1) / (frequency + normalization)
        results = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return results[:k]

    @staticmethod
    def dcg(grades: list[int]) -> float:
        return sum(
            (2 ** grade - 1) / math.log2(position + 1)
            for position, grade in enumerate(grades, start=1)
        )

    @staticmethod
    def validate_cuts(cuts):
        if not 1 <= len(cuts) <= 3:
            raise ValueError("Ingresa de uno a tres cortes.")
        if any(type(cut) is not int or cut <= 0 for cut in cuts):
            raise ValueError("Cada corte debe ser un entero positivo.")
        if len(set(cuts)) != len(cuts):
            raise ValueError("Los cortes deben ser distintos.")

    @staticmethod
    def ranking_profile(ranking, judgments):
        # Solo guardar posiciones relevantes: permite cambiar k sin repetir la
        # búsqueda ni retener todos los rankings del corpus en memoria.
        relevant = [(position, judgments[str(doc_id)])
                    for position, (doc_id, _) in enumerate(ranking, start=1)
                    if judgments.get(str(doc_id), 0) >= 1]
        return {"size": len(ranking), "relevant": relevant,
                "ideal": sorted((g for g in judgments.values() if g >= 1), reverse=True)}

    def evaluate_profile(self, profile, cuts):
        relevant, ideal = profile["relevant"], profile["ideal"]
        total = len(ideal)
        found = len(relevant)
        result = {
            "AP": sum(hit / position for hit, (position, _) in enumerate(relevant, 1)) / total if total else 0.0,
            "Precision": found / profile["size"] if profile["size"] else 0.0,
            "Recall": found / total if total else 0.0,
        }
        for cut in cuts:
            selected = [(position, grade) for position, grade in relevant if position <= cut]
            ideal_dcg = self.dcg(ideal[:cut])
            result[f"Precision@{cut}"] = len(selected) / cut
            result[f"Recall@{cut}"] = len(selected) / total if total else 0.0
            result[f"NDCG@{cut}"] = (
                sum((2 ** grade - 1) / math.log2(position + 1) for position, grade in selected)
                / ideal_dcg if ideal_dcg else 0.0
            )
        return result

    def evaluate_ranking(self, ranking, judgments, cuts=(5, 10, 20)):
        self.validate_cuts(cuts)
        return self.evaluate_profile(self.ranking_profile(ranking, judgments), cuts)

    def evaluate_cuts(self, cuts: tuple[int, ...]) -> dict:
        self.validate_cuts(cuts)
        with self.lock:
            profiles = dict(self.evaluation_profiles)
        per_query = {
            name: {qid: self.evaluate_profile(profile, cuts) for qid, profile in queries.items()}
            for name, queries in profiles.items()
        }
        summary = {}
        for name, detail in per_query.items():
            columns = ["AP", "Precision", "Recall"] + [
                f"{metric}@{cut}" for cut in cuts for metric in ("Precision", "Recall", "NDCG")
            ]
            summary[name] = {
                "MAP" if column == "AP" else column:
                sum(row[column] for row in detail.values()) / len(detail) if detail else 0.0
                for column in columns
            }
        return {"summary": summary, "per_query": per_query}

    def ensure_dense_evaluation(self):
        with self.lock:
            if (not self.evaluation_ready or self.dense_evaluation_loading
                    or self.dense_evaluation_error or "Embeddings" in self.evaluation_profiles):
                return
            self.dense_evaluation_loading = True
            generation = self.evaluation_generation
            documents = self.documents
            classified = self.evaluation_classified
        threading.Thread(target=self._evaluate_dense, args=(generation, documents, classified), daemon=True).start()

    def _evaluate_dense(self, generation, documents, classified):
        try:
            while True:
                with self.lock:
                    if generation != self.evaluation_generation:
                        return
                    if self.dense.ensure_ready(documents):
                        break
                    status = self.dense.status()
                if status["state"] == "error":
                    raise RuntimeError(status["error"])
                time.sleep(0.2)
            profiles = {}
            for qid, data in classified.items():
                if generation != self.evaluation_generation:
                    return
                ranking = self.dense.search(data["text"], len(documents))
                profiles[qid] = self.ranking_profile(ranking, data["judgments"])
                with self.lock:
                    if generation != self.evaluation_generation:
                        return
                    self.dense_evaluation_profiles[qid] = profiles[qid]
            with self.lock:
                if generation == self.evaluation_generation:
                    self.evaluation_profiles["Embeddings"] = profiles
        except Exception as exc:
            with self.lock:
                if generation == self.evaluation_generation:
                    self.dense_evaluation_error = str(exc)
        finally:
            with self.lock:
                if generation == self.evaluation_generation:
                    self.dense_evaluation_loading = False

    def classify_query_type(self, text: str) -> str:
        patterns = (
            ("comparacion", re.compile(r"\b(?:difference[s]? between|compar(?:e|ed|ison)|better than|versus|vs)\b", re.I)),
            ("definicion", re.compile(r"^\s*what (?:(?:is|are)\b|does .+ mean\b)|\bmeaning of\b", re.I)),
            ("procedimiento", re.compile(r"^\s*how (?:do|can|should|could) (?:i|we|you|one)\b|^\s*how to\b", re.I)),
            ("explicacion", re.compile(r"^\s*(?:why\b|how (?:does|do|did)\b)", re.I)),
        )
        for query_type, pattern in patterns:
            if pattern.search(text):
                return query_type
        return "otras"

    def load_fiqa_evaluation(self) -> None:
        with self.lock:
            if self.evaluation_loading:
                return
            self.evaluation_loading = True
            self.evaluation_ready = False
            self.evaluation_error = ""
            generation = self.evaluation_generation
        try:
            from datasets import load_dataset

            queries = load_dataset("BeIR/fiqa", "queries", split="queries")
            qrels = load_dataset("BeIR/fiqa-qrels", split="test")

            query_text = {
                str(row["_id"]): str(row["text"])
                for row in queries
            }
            judgments = defaultdict(dict)
            for row in qrels:
                judgments[str(row["query-id"])][str(row["corpus-id"])] = int(row["score"])

            classified = {}
            lengths = []
            for query_id, query_judgments in sorted(judgments.items()):
                text = query_text.get(query_id)
                if text is None:
                    continue
                token_count = len(self.tokenize(text))
                length = token_count
                lengths.append(length)
                classified[query_id] = {
                    "text": text,
                    "judgments": query_judgments,
                    "token_count": length,
                    "query_type": self.classify_query_type(text),
                }

            lengths.sort()
            count = len(lengths)
            short_cut = lengths[(count + 2) // 3 - 1] if count else 0
            medium_cut = lengths[(2 * count + 2) // 3 - 1] if count else 0
            groups = {
                "length": {"cortas": [], "medianas": [], "largas": []},
                "type": {"comparacion": [], "definicion": [], "procedimiento": [], "explicacion": [], "otras": []},
            }
            for query_id, data in classified.items():
                if data["token_count"] <= short_cut:
                    length_group = "cortas"
                elif data["token_count"] <= medium_cut:
                    length_group = "medianas"
                else:
                    length_group = "largas"
                data["length_group"] = length_group
                groups["length"][length_group].append(query_id)
                groups["type"][data["query_type"]].append(query_id)

            profiles = {}
            rankers = {
                "Jaccard": self.ranking_jaccard,
                "TF-IDF": self.ranking_tfidf,
                "BM25": self.ranking_bm25,
            }
            for name, ranker in rankers.items():
                profiles[name] = {}
                for query_id, data in classified.items():
                    with self.lock:
                        if generation != self.evaluation_generation:
                            return
                        ranking = ranker(data["text"], len(self.documents))
                    profiles[name][query_id] = self.ranking_profile(ranking, data["judgments"])

            with self.lock:
                if generation != self.evaluation_generation:
                    return
                self.evaluation_ready = True
                self.evaluation_profiles = profiles
                evaluated = self.evaluate_cuts((5, 10, 20))
                summary, per_query = evaluated["summary"], evaluated["per_query"]
                self.evaluation_summary = summary
                self.evaluation_classified = classified
                self.evaluation_groups = {
                    "cuts": {"short": short_cut, "medium": medium_cut},
                    "length": groups["length"],
                    "type": groups["type"],
                }
                self.evaluation_queries = {
                    query_id: {
                        "text": data["text"],
                        "token_count": data["token_count"],
                        "length_group": data["length_group"],
                        "query_type": data["query_type"],
                        "judgment_count": len(data["judgments"]),
                        "metrics": {
                            model_name: per_query[model_name][query_id]
                            for model_name in per_query
                            if query_id in per_query[model_name]
                        },
                    }
                    for query_id, data in classified.items()
                }
        except Exception as exc:
            with self.lock:
                if generation == self.evaluation_generation:
                    self.evaluation_ready = False
                    self.evaluation_error = str(exc)
        finally:
            with self.lock:
                if generation == self.evaluation_generation:
                    self.evaluation_loading = False

    def format_hits(self, results: list[tuple[str, float]], query: str) -> list[dict]:
        hits = []
        terms = set(self.tokenize(query))
        for rank, (doc_id, score) in enumerate(results, start=1):
            document = self.get_document(doc_id)
            if not document:
                continue
            text = " ".join(
                value for value in (document["title"], document["text"]) if value
            )
            spans = []
            for match in re.finditer(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*", text):
                if match.group(0).lower() in terms:
                    spans.append([match.start(), match.end()])
            hits.append({
                "rank": rank,
                "doc_id": doc_id,
                "text": text,
                "score": round(float(score), 6),
                "matched": sorted(terms),
                "spans": spans,
            })
        return hits

    def search(self, query: str, model: str, k: int) -> list[dict]:
        with self.lock:
            if not self.ready:
                return []
            if model == "jaccard":
                return self.search_jaccard(query, k)
            if model == "tfidf":
                return self.search_tfidf(query, k)
            if model == "bm25":
                return self.search_bm25(query, k)
            if model == "dense":
                return self.search_dense(query, k)
            return []

    def suggestion(self, query: str) -> str:
        if not self.ready:
            return ""
        terms = set(self.tokenize(query))
        suggestions = []
        for term in terms:
            nearest = None
            nearest_distance = 4
            for vocabulary_term in self.tfidf_idf:
                if abs(len(vocabulary_term) - len(term)) > nearest_distance:
                    continue
                distance = self.levenshtein(term, vocabulary_term)
                if distance < nearest_distance:
                    nearest = vocabulary_term
                    nearest_distance = distance
            if nearest:
                suggestions.append(nearest)
        if not suggestions:
            return ""
        return " ".join(suggestions)

    @staticmethod
    def levenshtein(left: str, right: str) -> int:
        previous = list(range(len(right) + 1))
        for i, left_char in enumerate(left, start=1):
            current = [i]
            for j, right_char in enumerate(right, start=1):
                current.append(min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (left_char != right_char),
                ))
            previous = current
        return previous[-1]


engine = SearchEngine()


@app.route("/")
def index():
    return send_from_directory(FRONT_DIR, "index.html")


@app.route("/<path:path>")
def static_files(path):
    target = ROOT / path
    if target.is_file():
        return send_from_directory(ROOT, path)
    target = FRONT_DIR / path
    if target.is_file():
        return send_from_directory(FRONT_DIR, path)
    return jsonify({"error": "Archivo no encontrado"}), 404


@app.route("/api/corpus/status")
def corpus_status():
    with engine.lock:
        return jsonify({
            "ready": engine.ready,
            "loading": engine.loading,
            "error": engine.error,
            "count": len(engine.documents),
            "source": engine.source,
            "dense": engine.dense_status(),
        })


@app.route("/api/corpus/import", methods=["POST"])
def import_corpus():
    if engine.loading:
        return jsonify({"error": "La importacion ya esta en progreso"}), 409

    uploaded = request.files.get("file")
    if uploaded:
        filename = uploaded.filename or ""
        suffix = Path(filename).suffix.lower()
        try:
            payload = uploaded.read()
            if suffix == ".json":
                data = json.loads(payload.decode("utf-8"))
                documents = data if isinstance(data, list) else data.get("documents", [])
            elif suffix == ".jsonl":
                documents = [
                    json.loads(line)
                    for line in payload.decode("utf-8").splitlines()
                    if line.strip()
                ]
            else:
                documents = [
                    {"_id": str(position), "title": "", "text": line.strip()}
                    for position, line in enumerate(payload.decode("utf-8").splitlines(), start=1)
                    if line.strip()
                ]
            if not documents:
                return jsonify({"error": "El archivo no contiene documentos"}), 400
        except Exception as exc:
            return jsonify({"error": f"No se pudo leer el corpus: {exc}"}), 400

        threading.Thread(
            target=engine.import_documents,
            args=(documents, f"Archivo: {filename}"),
            daemon=True,
        ).start()
        return jsonify({"loading": True, "source": filename})

    payload = request.get_json(silent=True) or {}
    dataset_name = str(payload.get("dataset") or "BeIR/fiqa")
    split = str(payload.get("split") or "corpus")
    threading.Thread(
        target=engine.import_huggingface,
        args=(dataset_name, split),
        daemon=True,
    ).start()
    return jsonify({"loading": True, "source": f"Hugging Face: {dataset_name}/{split}"})


@app.route("/api/search", methods=["POST"])
def search():
    payload = request.get_json(silent=True) or {}
    query = str(payload.get("q") or "").strip()
    model = str(payload.get("model") or "jaccard").lower()
    try:
        k = min(max(int(payload.get("k") or 10), 1), 100)
    except (TypeError, ValueError):
        k = 10
    seq = int(payload.get("seq") or 0)
    started = time.perf_counter()

    if not engine.ready:
        status = "cargando..." if engine.loading else (
            f"Error al importar: {engine.error}" if engine.error
            else "Corpus no importado"
        )
        return jsonify({
            "seq": seq,
            "key": f"{model}|{k}|{query}",
            "model": model,
            "elapsed_ms": (time.perf_counter() - started) * 1000,
            "terms": engine.tokenize(query),
            "suggestion": "",
            "hits": [],
            "error": status,
        })

    if model == "dense" and not engine.dense_ready():
        dense_status = engine.dense_status()
        return jsonify({
            "seq": seq,
            "key": f"{model}|{k}|{query}",
            "model": model,
            "elapsed_ms": (time.perf_counter() - started) * 1000,
            "terms": engine.tokenize(query),
            "suggestion": "",
            "hits": [],
            "error": dense_status["message"],
            "dense": dense_status,
        })

    hits = engine.search(query, model, k)
    return jsonify({
        "seq": seq,
        "key": f"{model}|{k}|{query}",
        "model": model,
        "elapsed_ms": (time.perf_counter() - started) * 1000,
        "terms": engine.tokenize(query),
        "suggestion": "",
        "hits": hits,
    })


@app.route("/api/compare", methods=["POST"])
def compare():
    payload = request.get_json(silent=True) or {}
    query = str(payload.get("q") or "").strip()
    try:
        k = min(max(int(payload.get("k") or 10), 1), 100)
    except (TypeError, ValueError):
        k = 10
    if not engine.ready:
        return jsonify({
            "error": "El corpus aun no esta listo",
            "loading": engine.loading,
            "results": {},
        })
    dense_ready = engine.dense_ready()
    results = {
        model: engine.search(query, model, k)
        for model in ("jaccard", "tfidf", "bm25")
    }
    results["dense"] = engine.search(query, "dense", k) if dense_ready else []
    relevant_ids = {
        hit["doc_id"]
        for hit in results["bm25"][:k]
    }
    metrics = {}
    for model, hits in results.items():
        retrieved_ids = [hit["doc_id"] for hit in hits[:k]]
        relevant_retrieved = len(set(retrieved_ids).intersection(relevant_ids))
        precision = relevant_retrieved / k
        recall = relevant_retrieved / len(relevant_ids) if relevant_ids else 0.0
        metrics[model] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
        }
    return jsonify({
        "query": query,
        "k": k,
        "reference": "bm25",
        "results": results,
        "metrics": metrics,
        "dense": engine.dense_status(),
    })


@app.route("/api/metrics", methods=["GET"])
def metrics():
    cuts_raw = request.args.get("cuts")
    try:
        cuts = tuple(int(value.strip()) for value in cuts_raw.split(",")) if cuts_raw is not None else (5, 10, 20)
        engine.validate_cuts(cuts)
    except (ValueError, TypeError) as exc:
        return jsonify({"cut_error": "Ingresa de uno a tres enteros positivos y distintos.", "error": str(exc)}), 400

    engine.ensure_dense_evaluation()
    with engine.lock:
        dynamic = engine.evaluate_cuts(cuts)
        if "Embeddings" not in dynamic["per_query"]:
            # Publicar cada consulta terminada sin presentar un MAP parcial
            # como si fuera el promedio global de toda la colección de prueba.
            dynamic["per_query"]["Embeddings"] = {
                qid: engine.evaluate_profile(profile, cuts)
                for qid, profile in engine.dense_evaluation_profiles.items()
            }
        queries = {
            qid: {**data, "metrics": {name: rows[qid] for name, rows in dynamic["per_query"].items() if qid in rows}}
            for qid, data in engine.evaluation_queries.items()
        }
        return jsonify({
            "available": engine.ready,
            "document_count": len(engine.documents),
            "term_count": len(engine.tfidf_idf),
            "source": engine.source,
            "evaluation_ready": engine.evaluation_ready,
            "evaluation_loading": engine.evaluation_loading,
            "evaluation_error": engine.evaluation_error,
            "dense_evaluation_loading": engine.dense_evaluation_loading,
            "dense_evaluation_error": engine.dense_evaluation_error,
            "dense_evaluation_progress": {
                "done": len(engine.dense_evaluation_profiles),
                "total": len(engine.evaluation_classified),
            },
            "dense": engine.dense_status(),
            "summary": dynamic["summary"],
            "dynamic_summary": dynamic["summary"],
            "dynamic_per_query": dynamic["per_query"],
            "dynamic_cuts": list(cuts),
            "cut_error": None,
            "groups": engine.evaluation_groups,
            "queries": queries,
            "analysis": build_evaluation_analysis(
                dynamic["per_query"], engine.evaluation_classified,
                engine.evaluation_groups, engine.evaluation_profiles,
            ),
        })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
