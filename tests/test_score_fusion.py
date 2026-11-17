import ast
import logging
from pathlib import Path
from types import SimpleNamespace
import time
import unittest

from src.online_pipeline.diverse_candidates import select_diverse_indices
from src.online_pipeline.score_fusion import weighted_rrf, normalized_score, unit_score
from test_manual_ocr_service import namespace


def candidate(idx, score=0.5):
    return dict(faiss_idx=idx, score=score)


class FusionTests(unittest.TestCase):
    def test_primary_branch_outweighs_both_secondary_branches(self):
        result = weighted_rrf([[candidate(1)], [candidate(2)], [candidate(2)]], [.7, .2, .1])
        self.assertEqual([c['faiss_idx'] for c in result], [1, 2])
        self.assertAlmostEqual(result[0]['score'], .7)
        self.assertAlmostEqual(result[1]['score'], .3)

    def test_theoretical_normalization_missing_and_duplicate_hits(self):
        result = weighted_rrf([[candidate(1), candidate(1)], [candidate(1)], [candidate(1)]], [.7, .2, .1])
        self.assertAlmostEqual(result[0]['score'], 1)
        self.assertEqual(weighted_rrf([[]], [.7]), [])
        single = weighted_rrf([[candidate(1), candidate(2)]], [.7, .2, .1])
        self.assertAlmostEqual(single[0]['score'], 1)
        self.assertAlmostEqual(single[1]['score'], 61 / 62)

    def test_invalid_weights_and_cosine_conversion(self):
        for weights in ([0], [-1], [float('nan')], []):
            with self.assertRaises(ValueError):
                weighted_rrf([[candidate(1)]], weights)
        for score, expected in [(-1, 0), (0, .5), (1, 1)]:
            self.assertEqual(normalized_score(candidate(1, score)), expected)

    def test_service_batches_variants_and_applies_fusion(self):
        service = namespace['SearchService'].__new__(namespace['SearchService'])
        calls = []
        service.encode_query_variants = lambda q: ('vectors', ['primary', 'action', 'tags'])
        def batch(vectors, top_k):
            calls.append((vectors, top_k))
            return [[candidate(1)], [candidate(2)], [candidate(2)]]
        service.retriever = SimpleNamespace(search_batch=batch)
        namespace['config'].EXPANSION_WEIGHTS = [.7, .2, .1]
        namespace['config'].RRF_RANK_CONSTANT = 60
        result = service.retrieve_text('query', top_k=10)
        self.assertEqual(calls, [('vectors', 10)])
        self.assertEqual(result[0]['faiss_idx'], 1)

    def test_encode_keeps_separate_vectors_and_primary_for_image_blending(self):
        service = namespace['SearchService'].__new__(namespace['SearchService'])
        variants = ['A bicycle racer in yellow', 'A cyclist riding', 'cyclist, yellow']
        service.expander = SimpleNamespace(expand=lambda q: variants)
        batches = []
        def encode(texts):
            batches.append(texts)
            return [[1, 0], [0, 1], [1, 1]]
        service.encoder = SimpleNamespace(encode_text_batch=encode)
        vectors, texts = service.encode_query_variants('input')
        self.assertEqual(vectors, [[1, 0], [0, 1], [1, 1]])
        self.assertEqual(texts, variants)
        self.assertEqual(batches, [variants])
        self.assertEqual(service.encode_and_expand_query('input')[0], [[1, 0]])

    def test_expansion_failure_uses_single_translation_branch(self):
        service = namespace['SearchService'].__new__(namespace['SearchService'])
        def fail(query):
            raise RuntimeError('test model unavailable')
        service.expander = SimpleNamespace(expand=fail)
        service._normalize_query_language = lambda q: 'A bicycle racer'
        service.encoder = SimpleNamespace(encode_text_batch=lambda texts: [[1, 0]] * len(texts))
        with self.assertLogs(namespace['logger'], level='WARNING'):
            vectors, texts = service.encode_query_variants('input')
        self.assertEqual(texts, ['A bicycle racer'])
        self.assertEqual(vectors, [[1, 0]])


class RerankTests(unittest.TestCase):
    def test_bounded_repeatable_reranking_and_failure_preserves_retrieval(self):
        source = Path(__file__).resolve().parents[1] / 'src/online_pipeline/vlm_reranker.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'VLMReranker')
        scope = dict(select_diverse_indices=select_diverse_indices, unit_score=unit_score, time=time, logger=logging.getLogger(__name__),
                     fetch_frame_image=lambda *a, **kw: object())
        exec(compile('from __future__ import annotations\n' + ast.unparse(node), str(source), 'exec'), scope)
        reranker = scope['VLMReranker'].__new__(scope['VLMReranker'])
        reranker.model = object()
        reranker.media_base = ''
        reranker.score_single_frame = lambda *a: .8
        original = [dict(score=0.0, entity={'video': 'v', 'frame_id': '1'})]
        result = reranker.rerank_candidates(original, 'query')
        self.assertEqual(original[0]['score'], 0)
        self.assertAlmostEqual(result[0]['score'], .65 * .8)
        self.assertEqual(result, reranker.rerank_candidates(result, 'query'))
        reranker.score_single_frame = lambda *a: None
        failed = reranker.rerank_candidates(result, 'query')
        self.assertEqual(failed[0]['score'], 0)
        self.assertFalse(failed[0]['vlm_verified'])

        # Select deeper diverse candidates, score only those, retain skipped frames.
        from test_diverse_candidates import Index, frame
        index = Index([[1, 0], [1, 0], [0, 1]])
        candidates = [dict(frame(i, i), score=.8 - i * .1) for i in range(3)]
        scored = []
        def score(*args):
            scored.append(1)
            return .9
        reranker.score_single_frame = score
        diverse = reranker.rerank_candidates(candidates, 'query', top_n=2, diversity_index=index)
        self.assertEqual(len(scored), 2)
        self.assertEqual([c['id'] for c in diverse], [0, 2, 1])
        self.assertEqual([c['vlm_selected'] for c in diverse], [True, True, False])
        self.assertEqual(diverse[-1]['score'], candidates[1]['score'])


if __name__ == '__main__':
    unittest.main()
