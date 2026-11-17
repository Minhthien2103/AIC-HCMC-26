"""Exercise real service methods without importing the GPU-loading ASGI app.

Run with: python -m unittest discover -s tests -p test_manual_ocr_service.py
"""
import ast
import logging
from pathlib import Path
from types import SimpleNamespace
import unittest
from src.online_pipeline.score_fusion import normalized_score, unit_score, weighted_rrf


tree = ast.parse((Path(__file__).resolve().parents[1] / 'src/server.py').read_text(encoding='utf-8'))
service_node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SearchService')
namespace = {'config': SimpleNamespace(ENABLE_OCR_SEARCH=True, OCR_BOOST_WEIGHT=0.35),
             'logger': logging.getLogger(__name__), 'normalized_score': normalized_score,
             'unit_score': unit_score, 'weighted_rrf': weighted_rrf}
exec(compile('from __future__ import annotations\n' + ast.unparse(service_node), 'SearchService', 'exec'), namespace)


class OCR:
    def __init__(self):
        self.keywords = None
        self.extracted = False

    def extract_deterministic_keywords(self, query):
        self.extracted = True
        return ['115'] if '115' in query else []

    def search_candidates(self, keywords, boost_score):
        self.keywords = keywords
        return {('V', '002'): {'boost': boost_score, 'matched_keyword': keywords[0]},
                ('outside', '999'): {'boost': boost_score, 'matched_keyword': keywords[0]}}


class ManualOCRTests(unittest.TestCase):
    def setUp(self):
        self.service = namespace['SearchService'].__new__(namespace['SearchService'])
        self.service.text_search = OCR()
        self.service.encode_and_expand_query = lambda q: (q, [])
        self.service.retrieve_text = lambda *a, **kw: self.rows()
        self.service.retriever = SimpleNamespace(search=lambda *a, **kw: self.rows())
        self.service.format_results = lambda rows, limit: rows[:limit]

    def rows(self):
        return [dict(video_id='V', keyframe_name='001', frame_id=10, score=0.6, score_space='unit'),
                dict(video_id='V', keyframe_name='002', frame_id=20, score=0.4, score_space='unit')]

    def test_manual_phrase_overrides_regex_without_adding_candidates(self):
        rows = self.service.apply_ocr_boost(self.rows(), '115', '  HTV7  ')
        self.assertEqual(self.service.text_search.keywords, ['htv7'])
        self.assertFalse(self.service.text_search.extracted)
        self.assertEqual([r['keyframe_name'] for r in rows], ['002', '001'])
        self.assertAlmostEqual(rows[0]['score'], 0.75 / 1.35)
        self.assertAlmostEqual(rows[1]['score'], 0.6 / 1.35)

    def test_blank_uses_regex_and_plain_visual_query_gets_no_boost(self):
        self.service.apply_ocr_boost(self.rows(), '115', '  ')
        self.assertEqual(self.service.text_search.keywords, ['115'])
        self.assertEqual(self.service.apply_ocr_boost(self.rows(), 'woman cycling'), self.rows())

    def test_missing_ocr_store_keeps_visual_results(self):
        self.service.text_search = None
        self.assertEqual(self.service.apply_ocr_boost(self.rows(), '115', 'HTV7'), self.rows())

    def test_comma_alternatives_trim_deduplicate_and_preserve_phrases(self):
        rows = self.service.apply_ocr_boost(self.rows(), 'visual query',
                                            ' 115, Ambulance, ,115, AMBULANCE, emergency vehicle, ')
        self.assertEqual(self.service.text_search.keywords, ['115', 'ambulance', 'emergency vehicle'])
        self.assertFalse(self.service.text_search.extracted)
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(rows[0]['score'], 0.75 / 1.35)

    def test_only_commas_does_not_boost_or_fall_back_to_regex(self):
        self.assertEqual(self.service.apply_ocr_boost(self.rows(), '115', ' , , '), self.rows())
        self.assertIsNone(self.service.text_search.keywords)
        self.assertFalse(self.service.text_search.extracted)

    def test_manual_ocr_applies_once_for_one_two_and_three_events(self):
        for queries in (['a'], ['a', 'b'], ['a', 'b', 'c']):
            with self.subTest(queries=queries):
                baseline = self.service.process_multi_event_query(queries)
                result = self.service.process_multi_event_query(queries, ocr_text='HTV7')
                scores = {r['keyframe_name']: r['score'] for r in baseline}
                self.assertEqual(len(result), len(baseline))
                for row in result:
                    self.assertAlmostEqual(row['score'],
                                           (scores[row['keyframe_name']] + (0.35 if row['keyframe_name'] == '002' else 0)) / 1.35)
                self.assertEqual(result, self.service.process_multi_event_query(queries, ocr_text='HTV7'))


if __name__ == '__main__':
    unittest.main()
