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
