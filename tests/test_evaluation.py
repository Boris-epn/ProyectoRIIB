import sys
import time
import types
import unittest
from unittest import mock

import server
from server import SearchEngine

DOCS = [
    {'_id': 'a', 'text': 'credit loan'},
    {'_id': 'b', 'text': 'credit card'},
    {'_id': 'c', 'text': 'bread'},
]
QUERIES = [{'_id': 'q1', 'text': 'credit'}, {'_id': 'q2', 'text': 'unknownword'}]
QRELS = [
    {'query-id': 'q1', 'corpus-id': 'a', 'score': 1},
    {'query-id': 'q1', 'corpus-id': 'c', 'score': 1},
    {'query-id': 'q2', 'corpus-id': 'c', 'score': 1},
]


def datasets_fixture(*args, **kwargs):
    return QRELS if args[0].endswith('-qrels') else QUERIES


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.engine = SearchEngine()
        self.engine.import_documents(DOCS, 'test')
        self.fake_datasets = types.SimpleNamespace(load_dataset=datasets_fixture)
        with mock.patch.dict(sys.modules, {'datasets': self.fake_datasets}):
            self.engine.load_fiqa_evaluation()
        self.assertTrue(self.engine.evaluation_ready, self.engine.evaluation_error)
        self.patcher = mock.patch.object(server, 'engine', self.engine)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.client = server.app.test_client()

    def test_metrics_math_and_map_independence(self):
        ranking = [('a', 4), ('b', 3), ('c', 2)]
        judgments = {'a': 1, 'c': 2, 'missing': 1}
        metrics = self.engine.evaluate_ranking(ranking, judgments, (1, 2, 10))
        self.assertEqual(metrics['Precision@1'], 1)
        self.assertEqual(metrics['Recall@1'], 1 / 3)
        self.assertEqual(metrics['Precision@2'], .5)
        self.assertEqual(metrics['Precision@10'], .2)
        self.assertEqual(metrics['Recall@10'], 2 / 3)
        self.assertAlmostEqual(metrics['AP'], (1 + 2 / 3) / 3)
        self.assertEqual(metrics['Precision'], 2 / 3)
        self.assertEqual(metrics['Recall'], 2 / 3)
        self.assertEqual(metrics['AP'], self.engine.evaluate_ranking(ranking, judgments, (100,))['AP'])
        self.assertTrue(all(value == 0 for value in self.engine.evaluate_ranking([], {}, (5,)).values()))

    def test_macro_summary_includes_queries_with_no_hits(self):
        result = self.engine.evaluate_cuts((1, 1000))
        for name in ('Jaccard', 'TF-IDF', 'BM25'):
            self.assertEqual(result['summary'][name]['Precision@1'], .5)
            self.assertEqual(result['summary'][name]['Recall@1'], .25)
            self.assertEqual(result['summary'][name]['Precision'], .25)
            self.assertEqual(result['summary'][name]['Recall'], .25)
            self.assertEqual(result['per_query'][name]['q2']['AP'], 0)

    def test_invalid_cut_requests_return_400(self):
        for cuts in ('', 'x', '1.5', '0', '-1', '2,2', '1,2,3,4', '1,,2', '1,'):
            with self.subTest(cuts=cuts):
                response = self.client.get('/api/metrics', query_string={'cuts': cuts})
                self.assertEqual(response.status_code, 400)
                self.assertTrue(response.get_json()['cut_error'])

    def test_api_uses_selected_cuts_in_summary_and_query(self):
        self.engine.dense_evaluation_error = 'test: unavailable'
        for cuts in ('1', '3,15', '7,11,1000'):
            result = self.client.get('/api/metrics', query_string={'cuts': cuts}).get_json()
            self.assertEqual(result['dynamic_cuts'], list(map(int, cuts.split(','))))
            self.assertIn('MAP', result['summary']['BM25'])
            for cut in result['dynamic_cuts']:
                self.assertIn(f'Precision@{cut}', result['summary']['BM25'])
                self.assertIn(f'Recall@{cut}', result['queries']['q1']['metrics']['BM25'])

    def test_cut_changes_reuse_profiles(self):
        with mock.patch.object(self.engine, 'ranking_bm25', side_effect=AssertionError('reranked')), \
             mock.patch.object(self.engine, 'ranking_tfidf', side_effect=AssertionError('reranked')), \
             mock.patch.object(self.engine, 'ranking_jaccard', side_effect=AssertionError('reranked')):
            first = self.engine.evaluate_cuts((1,))
            second = self.engine.evaluate_cuts((40, 1000))
        self.assertEqual(first['summary']['BM25']['MAP'], second['summary']['BM25']['MAP'])

    def test_dense_evaluation_publishes_fourth_model(self):
        dense = mock.Mock()
        dense.ensure_ready.return_value = True
        dense.search.return_value = [('c', .8), ('a', .7), ('b', .1)]
        dense.status.return_value = {'state': 'ready'}
        self.engine.dense = dense
        self.engine.ensure_dense_evaluation()
        deadline = time.monotonic() + 5
        while self.engine.dense_evaluation_loading and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertFalse(self.engine.dense_evaluation_loading)
        self.assertEqual(self.engine.dense_evaluation_error, '')
        response = self.client.get('/api/metrics?cuts=1,3').get_json()
        self.assertEqual(set(response['summary']), {'Jaccard', 'TF-IDF', 'BM25', 'Embeddings'})
        self.assertEqual(response['summary']['Embeddings']['Recall@3'], 1)
        self.assertEqual(response['queries']['q1']['metrics']['Embeddings']['Precision@1'], 1)
        self.assertEqual(dense.search.call_count, 2)

    def test_dense_failure_preserves_lexical_metrics(self):
        dense = mock.Mock()
        dense.ensure_ready.return_value = False
        dense.status.return_value = {'state': 'error', 'error': 'Missing model'}
        self.engine.dense = dense
        self.engine._evaluate_dense(self.engine.evaluation_generation, self.engine.documents, self.engine.evaluation_classified)
        self.assertEqual(self.engine.dense_evaluation_error, 'Missing model')
        self.assertEqual(set(self.engine.evaluate_cuts((5,))['summary']), {'Jaccard', 'TF-IDF', 'BM25'})

    def test_reimport_clears_evaluation_and_stale_worker(self):
        old_generation = self.engine.evaluation_generation
        old_documents = self.engine.documents
        old_classified = self.engine.evaluation_classified
        self.engine.import_documents([{'_id': 'new', 'text': 'different'}], 'new')
        self.assertFalse(self.engine.evaluation_ready)
        self.assertFalse(self.engine.evaluation_profiles)
        self.assertFalse(self.engine.evaluation_queries)
        self.engine._evaluate_dense(old_generation, old_documents, old_classified)
        self.assertFalse(self.engine.evaluation_profiles)


if __name__ == '__main__':
    unittest.main()
