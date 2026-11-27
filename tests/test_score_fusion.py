import ast
from contextlib import closing
from src.online_pipeline.image_prefetch import prefetch_images
import logging
from pathlib import Path
from types import SimpleNamespace
import time
import unittest
from unittest.mock import patch
from src.online_pipeline.query_preparation import prepare_english, prepare_chunks

from src.online_pipeline.diverse_candidates import select_diverse_indices
from src.online_pipeline.score_fusion import (
    weighted_rrf, normalized_score, unit_score, temporal_neighborhood_boost,
    score_local_video_frames,
)
from test_manual_ocr_service import namespace


def candidate(idx, score=0.5):
    return dict(faiss_idx=idx, score=score)


class FusionTests(unittest.TestCase):
    def setUp(self):
        namespace.update(prepare_english=prepare_english, prepare_chunks=prepare_chunks)
        namespace['config'].EXPANSION_WEIGHTS = [.7, .2, .1]
        namespace['config'].RRF_RANK_CONSTANT = 60

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
        service.query_preparations = {'query': {'weights': [.7, .2, .1]}}
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
        service.encoder = SimpleNamespace(encode_text_batch=encode, text_context_length=64, text_token_count=lambda text: len(text.split()) + 2)
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
        service.encoder = SimpleNamespace(encode_text_batch=lambda texts: [[1, 0]] * len(texts), text_context_length=64, text_token_count=lambda text: len(text.split()) + 2)
        with patch('src.online_pipeline.query_preparation.translate_vi_to_en', return_value='A bicycle racer'), self.assertLogs('src.online_pipeline.query_preparation', level='WARNING'):
            vectors, texts = service.encode_query_variants('input')
        self.assertEqual(texts, ['A bicycle racer'])
        self.assertEqual(vectors, [[1, 0]])

    def test_temporal_neighborhood_boost_promotes_clustered_scene_over_outlier(self):
        candidates = [
            dict(video_id='V_OUTLIER', keyframe_name='010', pts_time=10.0, score=0.50, score_space='unit'),
            dict(video_id='V_SCENE', keyframe_name='040', pts_time=40.0, score=0.46, score_space='unit'),
            dict(video_id='V_SCENE', keyframe_name='041', pts_time=41.0, score=0.45, score_space='unit'),
            dict(video_id='V_SCENE', keyframe_name='042', pts_time=42.0, score=0.44, score_space='unit'),
            dict(video_id='V_FAR', keyframe_name='090', pts_time=90.0, score=0.49, score_space='unit'),
            dict(video_id='V_FAR', keyframe_name='120', pts_time=120.0, score=0.48, score_space='unit'),
        ]
        boosted = temporal_neighborhood_boost(candidates, window_seconds=6.0, neighbor_weight=0.25, max_neighbors=2)
        self.assertEqual(boosted[0]['video_id'], 'V_SCENE')
        self.assertEqual(boosted[0]['keyframe_name'], '041')
        outlier = next(c for c in boosted if c['video_id'] == 'V_OUTLIER')
        far = next(c for c in boosted if c['video_id'] == 'V_FAR' and c['keyframe_name'] == '090')
        self.assertAlmostEqual(outlier['score'], 0.50 / 1.25)
        self.assertAlmostEqual(far['score'], 0.49 / 1.25)
        self.assertTrue(all(0.0 <= c['score'] <= 1.0 for c in boosted))

    def test_temporal_neighborhood_boost_caps_dense_neighbors_and_handles_missing_time(self):
        dense = [dict(video_id='V_DENSE', keyframe_name=f'{i:03d}', pts_time=10.0 + i * 0.1, score=0.40, score_space='unit')
                 for i in range(10)]
        sparse = [dict(video_id='V_SPARSE', keyframe_name='001', pts_time=10.0, score=0.40, score_space='unit'),
                  dict(video_id='V_SPARSE', keyframe_name='002', pts_time=10.1, score=0.40, score_space='unit'),
                  dict(video_id='V_SPARSE', keyframe_name='003', pts_time=10.2, score=0.40, score_space='unit'),
                  dict(video_id='V_NOTIME', keyframe_name='001', pts_time=None, score=0.30, score_space='unit')]
        result = temporal_neighborhood_boost(dense + sparse, window_seconds=6.0, neighbor_weight=0.25, max_neighbors=2)
        self.assertAlmostEqual(result[0]['score'],
                               next(c['score'] for c in result if c['video_id'] == 'V_SPARSE'))
        notime = next(c for c in result if c['video_id'] == 'V_NOTIME')
        self.assertAlmostEqual(notime['score'], 0.30 / 1.25)

    def test_temporal_query_rescues_missing_r2_via_local_video_search(self):
        service = namespace['SearchService'].__new__(namespace['SearchService'])
        service.query_preparations = {'event2': {'weights': [0.7, 0.2, 0.1]}}
        service.encode_query_variants = lambda q: ('vecs2', ['v1', 'v2', 'v3'])
        service.format_results = lambda rows, limit: rows[:limit]
        service.apply_ocr_boost = lambda rows, q, ocr='': rows

        r1 = [
            dict(video_id='V_BOTH_GLOBAL', keyframe_name='001', frame_id=100, pts_time=4.0, score=0.75, score_space='unit'),
            dict(video_id='V_ONLY_R1', keyframe_name='001', frame_id=100, pts_time=4.0, score=0.90, score_space='unit'),
            dict(video_id='V_LOCAL_R2', keyframe_name='001', frame_id=100, pts_time=4.0, score=0.80, score_space='unit'),
        ]
        r2 = [
            dict(video_id='V_BOTH_GLOBAL', keyframe_name='005', frame_id=200, pts_time=8.0, score=0.70, score_space='unit'),
            # Out of forward window (before frame1): should trigger local rescue check.
            dict(video_id='V_LOCAL_R2', keyframe_name='000', frame_id=50, pts_time=2.0, score=0.60, score_space='unit'),
        ]
        service.retrieve_text = lambda q, top_k=1000: r1 if q == 'event1' else r2

        searched_videos = []
        def search_in_video_batch(vecs, video_id, top_k):
            searched_videos.append(video_id)
            if video_id == 'V_LOCAL_R2':
                return [
                    [dict(video_id='V_LOCAL_R2', keyframe_name='004', frame_id=150, pts_time=6.0, score=0.25)],
                    [dict(video_id='V_LOCAL_R2', keyframe_name='004', frame_id=150, pts_time=6.0, score=0.20)],
                    [dict(video_id='V_LOCAL_R2', keyframe_name='004', frame_id=150, pts_time=6.0, score=0.20)],
                ]
            # Unrelated video has only low noise cosine <= 0.05, should not boost.
            return [
                [dict(video_id=video_id, keyframe_name='002', frame_id=120, pts_time=4.8, score=0.03)],
                [dict(video_id=video_id, keyframe_name='002', frame_id=120, pts_time=4.8, score=0.02)],
                [dict(video_id=video_id, keyframe_name='002', frame_id=120, pts_time=4.8, score=0.02)],
            ]

        service.retriever = SimpleNamespace(search_in_video_batch=search_in_video_batch)
        result = service.process_temporal_query('event1', 'event2', top_k=10)
        # V_BOTH_GLOBAL already matched global r2, so only V_ONLY_R1 and V_LOCAL_R2 needed local search.
        self.assertEqual(searched_videos, ['V_ONLY_R1', 'V_LOCAL_R2'])
        self.assertEqual([r['video_id'] for r in result], ['V_BOTH_GLOBAL', 'V_LOCAL_R2', 'V_ONLY_R1'])
        self.assertAlmostEqual(result[2]['score'], 0.45)

    def test_score_local_video_frames_filters_noise_and_bounds_score(self):
        rows = score_local_video_frames(
            [[dict(video_id='V', frame_id=1, score=0.35)], [dict(video_id='V', frame_id=1, score=0.35)]],
            [0.7, 0.3],
            scale=0.40,
        )
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]['score'], 0.40)
        self.assertEqual(
            score_local_video_frames([[dict(video_id='V', frame_id=1, score=0.04)]], [1.0], scale=0.40),
            [],
        )


class RerankTests(unittest.TestCase):
    def test_bounded_repeatable_reranking_and_failure_preserves_retrieval(self):
        source = Path(__file__).resolve().parents[1] / 'src/online_pipeline/vlm_reranker.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'VLMReranker')
        scope = dict(closing=closing, prefetch_images=prefetch_images, select_diverse_indices=select_diverse_indices, unit_score=unit_score, time=time, logger=logging.getLogger(__name__),
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
        serial = reranker.rerank_candidates(candidates, 'query', top_n=2,
                                           diversity_index=index, image_workers=1)
        self.assertEqual(diverse, serial)


if __name__ == '__main__':
    unittest.main()
