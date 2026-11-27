"""Bounded retrieval scores, independent of GPU/model dependencies.

These are ranking utilities, not calibrated probabilities of relevance.
"""
import math


def unit_score(value):
    value = float(value)
    return max(0.0, min(1.0, value)) if math.isfinite(value) else 0.0


def normalized_score(candidate):
    score = float(candidate.get('score', 0.0))
    if candidate.get('score_space') == 'unit':
        return unit_score(score)
    # Existing FAISS paths use normalized SigLIP inner products in [-1, 1].
    return unit_score((score + 1.0) / 2.0)


def weighted_rrf(rankings, weights, rank_constant=60):
    if not rankings:
        return []
    if len(weights) < len(rankings):
        raise ValueError('Provide one RRF weight per query branch')
    weights = [float(w) for w in weights[:len(rankings)]]
    if any(not math.isfinite(w) or w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError('RRF weights must be finite, nonnegative, with positive sum')
    if not math.isfinite(rank_constant) or rank_constant < 0:
        raise ValueError('RRF rank constant must be finite and nonnegative')
    total = sum(weights)
    fused = {}
    for ranking, weight in zip(rankings, weights):
        seen = set()
        for rank, candidate in enumerate(ranking, 1):
            key = candidate.get('faiss_idx')
            if key is None:
                key = (candidate.get('video_id'), str(candidate.get('keyframe_name', candidate.get('frame_id'))))
            if key in seen or weight == 0:
                continue
            seen.add(key)
            if key not in fused:
                fused[key] = dict(candidate, score=0.0, score_space='unit')
            # Divide by the theoretical maximum, not the observed batch maximum.
            fused[key]['score'] += (weight / total) * (rank_constant + 1) / (rank_constant + rank)
    return sorted(fused.values(), key=lambda c: -c['score'])


def temporal_neighborhood_boost(candidates, window_seconds=6.0, neighbor_weight=0.25, max_neighbors=2):
    """Promote frames supported by nearby high-scoring frames in the same video."""
    if (not candidates or not math.isfinite(window_seconds) or window_seconds <= 0
            or not math.isfinite(neighbor_weight) or neighbor_weight <= 0 or max_neighbors < 1):
        return candidates

    items = [dict(c, score=normalized_score(c), score_space='unit') for c in candidates]
    by_video = {}
    for idx, item in enumerate(items):
        vid = str(item.get('video_id') or '').strip()
        try:
            stamp = float(item.get('pts_time'))
        except (TypeError, ValueError):
            continue
        if vid and math.isfinite(stamp):
            by_video.setdefault(vid, []).append((idx, stamp, item['score']))

    neighborhoods = [0.0] * len(items)
    for group in by_video.values():
        if len(group) < 2:
            continue
        for idx, stamp, _base in group:
            supports = []
            for other_idx, other_stamp, other_score in group:
                if other_idx == idx:
                    continue
                delta = abs(other_stamp - stamp)
                if delta <= window_seconds:
                    supports.append(other_score * (1.0 - delta / window_seconds))
            if supports:
                supports.sort(reverse=True)
                neighborhoods[idx] = sum(supports[:max_neighbors]) / max_neighbors

    for item, neighborhood in zip(items, neighborhoods):
        item['score'] = unit_score((item['score'] + neighbor_weight * neighborhood) / (1.0 + neighbor_weight))
    return [item for _, item in sorted(enumerate(items), key=lambda pair: (-pair[1]['score'], pair[0]))]


def score_local_video_frames(per_variant_rankings, weights, scale=0.40, cosine_floor=0.05, cosine_span=0.25):
    """Combine per-variant in-video cosine scores into a conservative [0, scale] range."""
    if not per_variant_rankings or not weights or scale <= 0 or cosine_span <= 0:
        return []
    branch_weights = [max(0.0, float(w)) for w in weights[:len(per_variant_rankings)]]
    total_weight = sum(branch_weights)
    if total_weight <= 0:
        return []
    branch_weights = [w / total_weight for w in branch_weights]

    merged = {}
    for ranking, weight in zip(per_variant_rankings, branch_weights):
        if weight == 0:
            continue
        seen = set()
        for candidate in ranking:
            key = candidate.get('faiss_idx')
            if key is None:
                key = (candidate.get('video_id'), str(candidate.get('frame_id', candidate.get('keyframe_name'))))
            if key in seen:
                continue
            seen.add(key)
            entry = merged.setdefault(key, {'row': dict(candidate), 'cos': 0.0})
            entry['cos'] += weight * float(candidate.get('score', 0.0))

    scored = []
    for entry in merged.values():
        local_unit = unit_score((entry['cos'] - cosine_floor) / cosine_span)
        if local_unit <= 0:
            continue
        row = entry['row']
        row['score'] = scale * local_unit
        row['score_space'] = 'unit'
        scored.append(row)
    return sorted(scored, key=lambda c: -c['score'])
