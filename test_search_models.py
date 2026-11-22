import unittest

from server import SearchEngine


class JaccardAndTfidfTests(unittest.TestCase):
    def setUp(self):
        self.engine = SearchEngine()
        self.engine.import_documents(
            [
                {"_id": "d1", "title": "", "text": "credit"},
                {"_id": "d2", "title": "", "text": "credit card loan bank"},
                {"_id": "d3", "title": "", "text": "mortgage"},
            ],
            "test",
        )

    def test_jaccard_notebook_example(self):
        results = self.engine.search_jaccard("credit card", 10)
        scores = {result["doc_id"]: result["score"] for result in results}
        self.assertEqual(scores, {"d1": 0.5, "d2": 0.5})

    def test_jaccard_query_repetition_is_ignored(self):
        normal = self.engine.search_jaccard("credit credit card", 10)
        repeated = self.engine.search_jaccard("credit card", 10)
        normal_scores = {result["doc_id"]: result["score"] for result in normal}
        repeated_scores = {result["doc_id"]: result["score"] for result in repeated}
        self.assertEqual(normal_scores, repeated_scores)

    def test_jaccard_uses_document_text_not_title(self):
        engine = SearchEngine()
        engine.import_documents(
            [
                {"_id": "d1", "title": "mortgage", "text": "credit"},
                {"_id": "d2", "title": "", "text": "credit card"},
            ],
            "title-test",
        )
        results = engine.search_jaccard("mortgage", 10)
        self.assertEqual(results, [])

    def test_tfidf_cosine_notebook_formula(self):
        results = self.engine.search_tfidf("credit card", 10)
        scores = {result["doc_id"]: result["score"] for result in results}
        self.assertIn("d1", scores)
        self.assertIn("d2", scores)
        self.assertGreater(scores["d2"], scores["d1"])

    def test_empty_and_unknown_query(self):
        self.assertEqual(self.engine.search_jaccard("the and", 10), [])
        self.assertEqual(self.engine.search_tfidf("xyzabc", 10), [])

    def test_bm25_notebook_example(self):
        engine = SearchEngine()
        engine.import_documents(
            [
                {"_id": "D1", "title": "", "text": "datos datos ciencia"},
                {"_id": "D2", "title": "", "text": "datos ciencia modelos aprendizaje"},
                {"_id": "D3", "title": "", "text": "ciencia modelos"},
            ],
            "bm25-test",
        )

        self.assertEqual(
            engine.document_lengths,
            {"D1": 3, "D2": 4, "D3": 2},
        )
        self.assertEqual(engine.average_document_length, 3)
        self.assertAlmostEqual(engine.bm25_idf["datos"], 0.4700, places=4)
        self.assertAlmostEqual(engine.bm25_idf["ciencia"], 0.13353139262452257, places=4)

        results = engine.search_bm25("datos ciencia", 10)
        ranking = [(hit["doc_id"], round(hit["score"], 4)) for hit in results]
        self.assertEqual(
            ranking,
            [("D1", 0.7798), ("D2", 0.5311), ("D3", 0.1546)],
        )
        self.assertEqual(
            engine.search_bm25("datos ciencia datos", 10),
            results,
        )
        self.assertEqual(engine.search_bm25("hipoteca", 10), [])


if __name__ == "__main__":
    unittest.main()
