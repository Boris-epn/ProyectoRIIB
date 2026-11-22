import unittest
from copy import deepcopy

from evaluation_analysis import build_evaluation_analysis, MODELS


class EvaluationAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.classified = {
            '1': {'text': 'What is a fund?', 'query_type': 'definicion'},
            '2': {'text': 'What is a mortgage?', 'query_type': 'definicion'},
            '3': {'text': 'How to invest?', 'query_type': 'procedimiento'},
            '4': {'text': 'Why fees?', 'query_type': 'explicacion'},
        }
        self.groups = {
            'type': {'definicion': ['1', '2'], 'procedimiento': ['3'], 'explicacion': ['4'], 'otras': []},
            'length': {'cortas': ['1', '3'], 'largas': ['2', '4']},
        }
        scores = {
            'Jaccard': [.1, .1, .1, 0], 'TF-IDF': [.4, .6, .9, 0],
            'BM25': [.3, .5, .5, 0], 'Embeddings': [.8, .2, .5, 0],
        }
        self.per_query = {name: {qid: {'AP': value} for qid, value in zip(self.classified, values)} for name, values in scores.items()}
        self.profiles = {name: {qid: {'relevant': [(2, 1)] if qid != '4' else []} for qid in self.classified} for name in MODELS}

    def analyze(self):
        return build_evaluation_analysis(self.per_query, self.classified, self.groups, self.profiles)

    def test_group_map_winners_ties_and_empty_groups(self):
        result = self.analyze()
        rows = {row['group']: row for row in result['by_group']['type']}
        self.assertEqual(rows['definicion']['count'], 2)
        self.assertEqual(rows['definicion']['scores']['Embeddings'], .5)
        self.assertEqual(rows['definicion']['winners'], ['TF-IDF', 'Embeddings'])
        self.assertEqual(rows['procedimiento']['winners'], ['TF-IDF'])
        self.assertEqual(rows['explicacion']['winners'], list(MODELS))
        self.assertEqual(rows['otras']['scores'], {})
        self.assertEqual(rows['otras']['winners'], [])
        self.assertEqual(result['best_types']['Embeddings']['groups'], ['definicion', 'procedimiento'])
        self.assertEqual(result['best_types']['TF-IDF']['groups'], ['procedimiento'])

    def test_semantic_counts_deltas_and_case_order(self):
        semantic = self.analyze()['semantic']
        self.assertTrue(semantic['ready'])
        bm25 = semantic['comparisons']['BM25']
        self.assertEqual((bm25['wins'], bm25['losses'], bm25['ties']), (1, 1, 2))
        self.assertAlmostEqual(bm25['mean_delta_ap'], .05)
        self.assertEqual(bm25['improvements'][0]['query_id'], '1')
        self.assertEqual(bm25['regressions'][0]['query_id'], '2')
        tfidf = semantic['comparisons']['TF-IDF']
        self.assertEqual((tfidf['wins'], tfidf['losses'], tfidf['ties']), (1, 2, 1))
        self.assertAlmostEqual(tfidf['mean_delta_ap'], -.1)
        self.assertEqual(tfidf['improvements'][0]['first_relevant_dense'], 2)
        self.assertEqual(sum(tfidf[key] for key in ('wins', 'losses', 'ties')), tfidf['count'])

    def test_partial_dense_evaluation_is_not_compared_to_full_models(self):
        del self.profiles['Embeddings']
        self.per_query['Embeddings'] = {'1': {'AP': .8}}
        result = self.analyze()
        self.assertFalse(result['semantic']['ready'])
        self.assertEqual(result['semantic']['comparisons'], {})
        self.assertNotIn('Embeddings', result['by_group']['type'][0]['scores'])

    def test_empty_data_has_no_invented_winners_or_cases(self):
        result = build_evaluation_analysis({}, {}, {}, {})
        self.assertEqual(result['best_types'], {})
        self.assertEqual(result['semantic']['comparisons'], {})

    def test_case_limit_no_improvements_and_no_mutation(self):
        self.per_query['Embeddings'] = deepcopy(self.per_query['BM25'])
        original = deepcopy(self.per_query)
        result = build_evaluation_analysis(self.per_query, self.classified, self.groups, self.profiles, case_limit=1)
        bm25 = result['semantic']['comparisons']['BM25']
        self.assertEqual(bm25['improvements'], [])
        self.assertEqual(bm25['regressions'], [])
        self.assertEqual(bm25['ties'], 4)
        self.assertEqual(self.per_query, original)
        self.assertLessEqual(len(result['semantic']['comparisons']['TF-IDF']['regressions']), 1)


if __name__ == '__main__':
    unittest.main()
