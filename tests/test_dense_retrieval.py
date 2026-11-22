import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import zlib
from pathlib import Path
from unittest import mock

import numpy as np

import server
from dense_retrieval import (
    DenseRetriever,
    SentenceTransformerEncoder,
    corpus_fingerprint,
    document_text,
    normalize_documents,
)
from server import SearchEngine


HAS_FAISS = importlib.util.find_spec("faiss") is not None
HAS_SENTENCE_TRANSFORMERS = importlib.util.find_spec("sentence_transformers") is not None


class FakeEncoder:
    """Codificador determinista sin descargas.

    Los sinónimos comparten dimensión ("concepto"), así que una consulta puede
    encontrar documentos sin compartir palabras, como haría un modelo real.
    """

    CONCEPTS = {
        "car": 0, "automobile": 0, "vehicle": 0,
        "loan": 1, "financing": 1, "credit": 1, "borrowed": 1,
        "stock": 2, "equity": 2, "shares": 2,
        "bread": 3, "bake": 3, "sourdough": 3,
    }
    DIM = 16

    def __init__(self):
        self.calls = []

    def encode(self, texts):
        self.calls.append(list(texts))
        vectors = np.zeros((len(texts), self.DIM), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in text.lower().split():
                word = word.strip(".,;:!?")
                if not word:
                    continue
                column = self.CONCEPTS.get(word)
                if column is None:
                    column = 4 + zlib.crc32(word.encode()) % (self.DIM - 4)
                vectors[row, column] += 1.0
        return vectors

    @property
    def corpus_calls(self):
        """Llamadas con más de un texto (las consultas van de una en una)."""
        return [call for call in self.calls if len(call) > 1]


class FailingEncoder(FakeEncoder):
    def encode(self, texts):
        if len(texts) > 1:
            raise AssertionError("no debería codificar el corpus: hay caché")
        return super().encode(texts)


DOCUMENTS = [
    {"_id": "d1", "title": "", "text": "I bought a vehicle with borrowed money"},
    {"_id": "d2", "title": "Equity", "text": "stock shares dividends"},
    {"_id": "d3", "title": "", "text": "bake sourdough bread at home"},
]


def wait_until(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class EncoderLoadingTests(unittest.TestCase):
    def test_cached_model_does_not_use_online_loading(self):
        constructor = mock.Mock(return_value=object())
        module = mock.Mock(SentenceTransformer=constructor)
        with mock.patch.dict(sys.modules, {"sentence_transformers": module}):
            encoder = SentenceTransformerEncoder(model_name="cached")
            loaded = encoder.load()
            self.assertIs(encoder.load(), loaded)
        constructor.assert_called_once_with("cached", device=None, local_files_only=True)

    def test_missing_model_falls_back_to_download(self):
        constructor = mock.Mock(side_effect=[OSError("not cached"), object()])
        module = mock.Mock(SentenceTransformer=constructor)
        with mock.patch.dict(sys.modules, {"sentence_transformers": module}):
            SentenceTransformerEncoder(model_name="missing").load()
        self.assertEqual(constructor.call_args_list, [
            mock.call("missing", device=None, local_files_only=True),
            mock.call("missing", device=None),
        ])


class DocumentPreparationTests(unittest.TestCase):
    def test_normalize_matches_server_cleaning(self):
        raw = [
            {"_id": "a", "title": "T", "text": "x"},
            {"id": "b", "text": "y"},
            {"_id": 0, "content": "z"},
            {"title": None, "text": None},
        ]
        engine = SearchEngine()
        engine.import_documents(raw, "test")
        self.assertEqual(normalize_documents(raw), engine.documents)

    def test_normalize_is_idempotent(self):
        once = normalize_documents([{"id": 7, "text": "x"}, {"text": "y"}])
        self.assertEqual(normalize_documents(once), once)

    def test_document_text_joins_title_and_text(self):
        self.assertEqual(document_text({"title": "T", "text": "body"}), "T body")
        self.assertEqual(document_text({"title": "", "text": "body"}), "body")

    def test_fingerprint_depends_on_content_order_and_model(self):
        docs = normalize_documents(DOCUMENTS)
        base = corpus_fingerprint(docs, "m")
        self.assertEqual(base, corpus_fingerprint(normalize_documents(DOCUMENTS), "m"))
        self.assertNotEqual(base, corpus_fingerprint(docs[::-1], "m"))
        self.assertNotEqual(base, corpus_fingerprint(docs, "otro"))
        changed = [dict(d) for d in docs]
        changed[0]["text"] += "!"
        self.assertNotEqual(base, corpus_fingerprint(changed, "m"))


@unittest.skipUnless(HAS_FAISS, "faiss-cpu no está instalado")
class FakeEncoderRetrieverTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cache_dir = Path(self.tmp.name)
        self.encoder = FakeEncoder()
        self.retriever = DenseRetriever(model_name="fake", cache_dir=self.cache_dir, encoder=self.encoder)

    def test_finds_synonyms_without_shared_words(self):
        self.retriever.build(DOCUMENTS)
        results = self.retriever.search("car loan", 3)
        self.assertEqual(results[0][0], "d1")
        self.assertGreater(results[0][1], results[1][1])

    def test_scores_are_cosines_sorted_descending(self):
        self.retriever.build(DOCUMENTS)
        results = self.retriever.search("stock bread", 10)
        self.assertEqual(len(results), 3)  # k mayor que el corpus
        scores = [score for _, score in results]
        self.assertEqual(scores, sorted(scores, reverse=True))
        for score in scores:
            self.assertGreaterEqual(score, -1.0 - 1e-6)
            self.assertLessEqual(score, 1.0 + 1e-6)

    def test_respects_k(self):
        self.retriever.build(DOCUMENTS)
        self.assertEqual(len(self.retriever.search("stock", 1)), 1)

    def test_empty_query_and_not_ready_return_nothing(self):
        self.assertEqual(self.retriever.search("car", 5), [])
        self.retriever.build(DOCUMENTS)
        self.assertEqual(self.retriever.search("   ", 5), [])

    def test_build_writes_cache_and_second_retriever_reads_it(self):
        status = self.retriever.build(DOCUMENTS)
        self.assertEqual(status["origin"], "built")
        meta = json.loads((self.retriever.cache_location / "meta.json").read_text("utf-8"))
        self.assertEqual(meta["count"], 3)
        self.assertEqual(meta["dim"], FakeEncoder.DIM)

        other = DenseRetriever(model_name="fake", cache_dir=self.cache_dir, encoder=FailingEncoder())
        self.assertEqual(other.build(DOCUMENTS)["origin"], "cache")
        self.assertEqual(other.search("car loan", 1)[0][0], "d1")

    def test_cache_not_reused_for_other_corpus(self):
        self.retriever.build(DOCUMENTS)
        changed = DOCUMENTS + [{"_id": "d4", "title": "", "text": "automobile credit"}]
        status = self.retriever.build(changed)
        self.assertEqual(status["origin"], "built")
        self.assertEqual(len(self.encoder.corpus_calls), 2)

    def test_corrupted_cache_is_rebuilt(self):
        self.retriever.build(DOCUMENTS)
        (self.retriever.cache_location / "ids.json").write_text('["x"]', encoding="utf-8")
        other = DenseRetriever(model_name="fake", cache_dir=self.cache_dir, encoder=FakeEncoder())
        self.assertEqual(other.build(DOCUMENTS)["origin"], "built")

    def test_background_preparation_becomes_ready(self):
        documents = list(DOCUMENTS)
        self.assertFalse(self.retriever.ensure_ready(documents))
        self.assertTrue(wait_until(lambda: self.retriever.ready))
        self.assertTrue(self.retriever.ensure_ready(documents))
        self.assertEqual(self.retriever.search("car loan", 1)[0][0], "d1")

    def test_new_document_list_invalidates_index(self):
        documents = list(DOCUMENTS)
        self.retriever.build(documents)
        self.assertTrue(self.retriever.ensure_ready(documents))
        self.assertFalse(self.retriever.ensure_ready(list(DOCUMENTS)))  # otra lista
        self.assertTrue(wait_until(lambda: self.retriever.ready))

    def test_missing_cache_without_build_reports_error(self):
        retriever = DenseRetriever(
            model_name="fake", cache_dir=self.cache_dir, encoder=FakeEncoder(), build_if_missing=False
        )
        self.assertFalse(retriever.ensure_ready(list(DOCUMENTS)))
        self.assertTrue(wait_until(lambda: retriever.state == "error"))
        self.assertIn("--build", retriever.status()["error"])

    def test_results_fit_format_hits(self):
        engine = SearchEngine()
        engine.import_documents(DOCUMENTS, "test")
        self.retriever.build(engine.documents)
        hits = engine.format_hits(self.retriever.search("car loan", 2), "car loan")
        self.assertEqual([hit["rank"] for hit in hits], [1, 2])
        self.assertEqual(hits[0]["doc_id"], "d1")
        self.assertEqual(
            set(hits[0]), {"rank", "doc_id", "text", "score", "matched", "spans"}
        )


class ServerConfigurationTests(unittest.TestCase):
    def test_build_if_missing_reads_environment(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DENSE_BUILD_IF_MISSING", None)
            self.assertTrue(SearchEngine().dense.build_if_missing)
        for value in ("0", "false", "OFF"):
            with mock.patch.dict(os.environ, {"DENSE_BUILD_IF_MISSING": value}):
                self.assertFalse(SearchEngine().dense.build_if_missing)
        with mock.patch.dict(os.environ, {"DENSE_BUILD_IF_MISSING": "1"}):
            self.assertTrue(SearchEngine().dense.build_if_missing)

    def test_dense_batch_size_can_limit_inference_memory(self):
        with mock.patch.dict(os.environ, {"DENSE_BATCH_SIZE": "8"}):
            self.assertEqual(SearchEngine().dense.encoder.batch_size, 8)
        with mock.patch.dict(os.environ, {"DENSE_BATCH_SIZE": "invalid"}):
            self.assertEqual(SearchEngine().dense.encoder.batch_size, 64)

    def test_server_does_not_import_heavy_libraries(self):
        code = (
            "import sys, server; server.SearchEngine(); "
            "print([m for m in ('sentence_transformers', 'torch', 'faiss') if m in sys.modules])"
        )
        output = subprocess.run(
            [sys.executable, "-c", code], cwd=Path(__file__).resolve().parent.parent,
            capture_output=True, text=True, check=True,
        ).stdout.strip().splitlines()[-1]
        self.assertEqual(output, "[]")

    def test_building_message_shows_progress(self):
        engine = SearchEngine()
        engine.dense.state, engine.dense.done, engine.dense.total = "building", 12000, 57638
        self.assertEqual(engine.dense_status()["message"], "Generando embeddings… 12 000/57 638")
        engine.dense.state = "loading"
        self.assertEqual(engine.dense_status()["message"], "Cargando índice semántico…")


@unittest.skipUnless(HAS_FAISS, "faiss-cpu no está instalado")
class ServerEndpointTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.engine = SearchEngine()
        self.engine.dense = DenseRetriever(model_name="fake", cache_dir=tmp.name, encoder=FakeEncoder())
        self.engine.import_documents(DOCUMENTS, "test")
        patcher = mock.patch.object(server, "engine", self.engine)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.client = server.app.test_client()

    def post(self, path, payload):
        return self.client.post(path, json=payload).get_json()

    def test_search_reports_progress_without_engine_lock(self):
        locked, release = threading.Event(), threading.Event()

        def hold_lock():
            with self.engine.lock:
                locked.set()
                release.wait(10)

        holder = threading.Thread(target=hold_lock, daemon=True)
        holder.start()
        self.assertTrue(locked.wait(5))
        try:
            # Otro hilo tiene engine.lock: si la petición lo necesitara, se colgaría.
            response = self.post("/api/search", {"q": "car loan", "model": "dense", "k": 3})
            self.assertEqual(response["hits"], [])
            self.assertTrue(response["error"])
            self.assertIn(response["dense"]["state"], {"loading", "building", "ready"})
            # La construcción en segundo plano avanza aunque engine.lock esté ocupado.
            self.assertTrue(wait_until(lambda: self.engine.dense.ready))
        finally:
            release.set()
            holder.join(5)

        response = self.post("/api/search", {"q": "car loan", "model": "dense", "k": 3})
        self.assertNotIn("error", response)
        self.assertEqual(response["hits"][0]["doc_id"], "d1")

    def test_compare_includes_dense_without_blocking_others(self):
        response = self.post("/api/compare", {"q": "stock", "k": 3})
        self.assertEqual(set(response["results"]), {"jaccard", "tfidf", "bm25", "dense"})
        self.assertEqual(response["results"]["dense"], [])
        self.assertTrue(response["results"]["bm25"])
        self.assertTrue(wait_until(lambda: self.engine.dense.ready))
        response = self.post("/api/compare", {"q": "stock", "k": 3})
        self.assertEqual(response["results"]["dense"][0]["doc_id"], "d2")

    def test_status_includes_dense(self):
        status = self.client.get("/api/corpus/status").get_json()
        self.assertEqual(status["dense"]["state"], "idle")  # consultar el estado no carga nada
        self.post("/api/search", {"q": "car", "model": "dense"})
        self.assertTrue(wait_until(lambda: self.engine.dense.ready))
        status = self.client.get("/api/corpus/status").get_json()
        self.assertEqual(status["dense"]["state"], "ready")
        self.assertEqual(status["dense"]["message"], "")


@unittest.skipUnless(
    HAS_FAISS and HAS_SENTENCE_TRANSFORMERS,
    "requiere faiss-cpu y sentence-transformers",
)
class RealModelTests(unittest.TestCase):
    def test_semantic_match_without_lexical_overlap(self):
        documents = [
            {"_id": "auto", "title": "", "text": "I bought a used vehicle and paid for it with a five year bank loan."},
            {"_id": "index", "title": "", "text": "Index funds track the S&P 500 with low fees."},
            {"_id": "bread", "title": "", "text": "How to bake sourdough bread at home."},
        ]
        query = "car financing"
        engine = SearchEngine()
        engine.import_documents(documents, "test")
        self.assertEqual(engine.search_jaccard(query, 10), [])  # sin palabras en común

        with tempfile.TemporaryDirectory() as tmp:
            retriever = DenseRetriever(cache_dir=tmp)
            try:
                retriever.build(engine.documents)
            except OSError as exc:  # p. ej. sin red para descargar el modelo
                self.skipTest(f"no se pudo cargar el modelo: {exc}")
            results = retriever.search(query, 3)
        self.assertEqual(results[0][0], "auto")


if __name__ == "__main__":
    unittest.main()
