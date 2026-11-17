"""Choose VLM candidates using cached FAISS vectors; never fetch or encode images."""
import logging
import time

import numpy as np

logger = logging.getLogger(__name__)


def select_diverse_indices(candidates, index, top_n=20, pool_size=100,
                           similarity_threshold=0.97, time_window=3.0):
    limit = min(len(candidates), max(0, int(top_n)))
    if not limit:
        return []
    if index is None:
        return list(range(limit))
    started = time.perf_counter()
    pool = candidates[:max(limit, int(pool_size))]
    positions, ids = [], []
    videos, times = [], []
    for pos, candidate in enumerate(pool):
        entity = candidate.get('entity') or {}
        videos.append(str(entity.get('video') or candidate.get('video_id') or ''))
        try:
            stamp = float(entity.get('time', candidate.get('pts_time')))
            times.append(stamp if np.isfinite(stamp) else np.nan)
        except (TypeError, ValueError):
            times.append(np.nan)
        try:
            idx = int(candidate.get('faiss_idx', candidate.get('id')))
            if 0 <= idx < index.ntotal:
                positions.append(pos)
                ids.append(idx)
        except (TypeError, ValueError):
            pass

    # Missing metadata/vector means "unknown", never sufficient to discard a frame.
    similarities = np.full((len(pool), len(pool)), -np.inf, dtype=np.float32)
    if ids:
        try:
            if hasattr(index, 'reconstruct_batch'):
                vectors = index.reconstruct_batch(np.asarray(ids, dtype=np.int64))
            else:
                vectors = np.stack([index.reconstruct(idx) for idx in ids])
            vectors = np.asarray(vectors, dtype=np.float32)
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            valid = np.isfinite(vectors).all(axis=1) & (norms[:, 0] > 0)
            vectors = vectors[valid] / norms[valid]
            valid_positions = np.asarray(positions)[valid]
            similarities[np.ix_(valid_positions, valid_positions)] = vectors @ vectors.T
        except Exception as exc:
            logger.warning('Diversity vector lookup failed; keeping retrieval order: %s', exc)
            return list(range(limit))

    stamps = np.asarray(times)
    video_ids = np.asarray(videos)
    same_video = (video_ids[:, None] == video_ids[None, :]) & (video_ids[:, None] != '')
    near = np.abs(stamps[:, None] - stamps[None, :]) <= time_window
    redundant = same_video & near & (similarities >= similarity_threshold)
    selected = []
    for pos in range(len(pool)):
        if selected and redundant[pos, selected].any():
            continue
        selected.append(pos)
        if len(selected) == limit:
            break
    logger.info('[VLM Diversity] Selected %d/%d from pool %d in %.2f ms',
                len(selected), limit, len(pool), (time.perf_counter() - started) * 1000)
    return selected
