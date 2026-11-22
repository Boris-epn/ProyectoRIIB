import json
import math
import re
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory


ROOT = Path(__file__).resolve().parent
FRONT_DIR = ROOT / "front"

app = Flask(__name__)


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
    results = {
        model: engine.search(query, model, k)
        for model in ("jaccard", "tfidf", "bm25")
    }
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
    })


@app.route("/api/metrics", methods=["GET"])
def metrics():
    return jsonify({
        "available": engine.ready,
        "document_count": len(engine.documents),
        "term_count": len(engine.tfidf_idf),
        "source": engine.source,
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
