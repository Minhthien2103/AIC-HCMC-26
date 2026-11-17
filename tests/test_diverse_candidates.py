import unittest

import numpy as np

from src.online_pipeline.diverse_candidates import select_diverse_indices


class Index:
    def __init__(self, vectors):
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.ntotal = len(vectors)
        self.calls = 0

    def reconstruct_batch(self, ids):
        self.calls += 1
        return self.vectors[ids]


def frame(idx, seconds, video='a'):
    return dict(id=idx, entity=dict(video=video, time=seconds))


class DiversityTests(unittest.TestCase):
    def test_skip_adjacent_duplicates_but_keep_scene_changes_and_distant_scenes(self):
        index = Index([[1, 0], [1, 0], [0, 1], [1, 0], [1, 0]])
        frames = [frame(0, 0), frame(1, 1), frame(2, 2), frame(3, 10), frame(4, 1, 'b')]
        self.assertEqual(select_diverse_indices(frames, index, top_n=4), [0, 2, 3, 4])
        self.assertEqual(index.calls, 1)

    def test_identical_pool_uses_fewer_vlm_slots_instead_of_refilling_duplicates(self):
        index = Index([[1, 0]] * 10)
        frames = [frame(i, i / 10) for i in range(10)]
        self.assertEqual(select_diverse_indices(frames, index, top_n=5), [0])

    def test_pool_limit_and_zero_budget(self):
        index = Index([[1, 0]] * 6 + [[0, 1]])
        frames = [frame(i, i / 10) for i in range(7)]
        self.assertEqual(select_diverse_indices(frames, index, top_n=2, pool_size=6), [0])
        self.assertEqual(select_diverse_indices(frames, index, top_n=0), [])

    def test_missing_vectors_or_timestamps_do_not_remove_frames(self):
        index = Index([[1, 0], [1, 0]])
        frames = [frame(0, 0), frame(1, None), frame(99, 0)]
        self.assertEqual(select_diverse_indices(frames, index, top_n=3), [0, 1, 2])

    def test_invalid_vectors_and_no_index_fall_back_safely(self):
        index = Index([[0, 0], [float('nan'), 0], [1, 0]])
        frames = [frame(i, 0) for i in range(3)]
        self.assertEqual(select_diverse_indices(frames, index, top_n=3), [0, 1, 2])
        self.assertEqual(select_diverse_indices(frames, None, top_n=2), [0, 1])

    def test_index_failure_keeps_retrieval_order(self):
        class BrokenIndex:
            ntotal = 100

            def reconstruct_batch(self, ids):
                raise RuntimeError('test unsupported index')
        with self.assertLogs('src.online_pipeline.diverse_candidates', level='WARNING'):
            result = select_diverse_indices([frame(0, 0), frame(1, 0)], BrokenIndex())
        self.assertEqual(result, [0, 1])


if __name__ == '__main__':
    unittest.main()
