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


if __name__ == "__main__":
    unittest.main()
