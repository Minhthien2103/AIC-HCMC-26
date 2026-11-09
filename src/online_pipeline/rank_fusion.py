"""Deterministic, inspectable rank fusion for KIS evidence sources.

Unlike score fusion, reciprocal-rank fusion (RRF) does not assume that CLIP,
OCR, text metadata and a VLM use comparable score scales.  Every enabled
source contributes exactly one rank-based vote and the returned candidates
retain the individual ranks for audit/review.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def candidate_identity(candidate: dict[str, Any]) -> str:
    """Return a stable key for a BTC keyframe candidate.

    ``faiss_idx`` is the authoritative identity for the supplied index.  The
    fallback keeps this helper useful for review fixtures and future indexes.
    """
    # A neighbourhood proposal may originate from one FAISS keyframe but
    # represents a different original MP4 frame.  Its submitted identity is
    # therefore video/frame, not the anchor's FAISS id.
    if candidate.get("anchor_frame_id") is not None or candidate.get("neighborhood_rank") is not None:
        return "frame:{video}:{frame}".format(
            video=str(candidate.get("video_id", "")),
            frame=str(candidate.get("frame_id", candidate.get("keyframe_name", ""))),
        )
    if candidate.get("events"):
        return "sequence:{video}:{frames}".format(
            video=str(candidate.get("video_id", "")),
            frames=":".join(str(event.get("frame_id", "")) for event in candidate["events"]),
        )
    if candidate.get("faiss_idx") is not None:
        return f"faiss:{int(candidate['faiss_idx'])}"
    return "frame:{video}:{frame}".format(
        video=str(candidate.get("video_id", "")),
        frame=str(candidate.get("frame_id", candidate.get("keyframe_name", ""))),
    )


def fuse_rankings(
    rankings: Iterable[tuple[str, list[dict[str, Any]]]],
    *,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """Fuse named ranked lists with equal-weight RRF.

    Source lists may have different lengths. Duplicate candidates in a source
    are ignored after their first occurrence.  ``source_ranks`` and
    ``source_scores`` expose all evidence rather than hiding it in a tuned
    coefficient.
    """
    if rrf_k < 0:
        raise ValueError("rrf_k must be non-negative")

    merged: dict[str, dict[str, Any]] = {}
    for source_name, candidates in rankings:
        name = str(source_name).strip()
        if not name:
            raise ValueError("Fusion source name cannot be empty")
        seen_source: set[str] = set()
        for position, candidate in enumerate(candidates, start=1):
            key = candidate_identity(candidate)
            if key in seen_source:
                continue
            seen_source.add(key)
            if key not in merged:
                item = candidate.copy()
                # A source may itself have fused query variants. Its internal
                # score belongs in ``source_scores``; it must not leak into
                # this outer RRF sum.
                item["rrf_score"] = 0.0
                merged[key] = item
            else:
                item = merged[key]
                # Keep source-specific audit evidence (OCR text, metadata,
                # Qwen constraints) from later lists without replacing the
                # already established canonical keyframe fields.
                for field, value in candidate.items():
                    item.setdefault(field, value)
            item.setdefault("source_ranks", {})[name] = position
            item.setdefault("source_scores", {})[name] = float(candidate.get("score", 0.0))
            item["rrf_score"] = float(item.get("rrf_score", 0.0)) + 1.0 / (rrf_k + position)

    output = list(merged.values())
    for item in output:
        item["score"] = float(item["rrf_score"])
        item["source_ranks"] = dict(sorted(item["source_ranks"].items()))
        item["source_scores"] = dict(sorted(item["source_scores"].items()))
    output.sort(
        key=lambda item: (
            -float(item["score"]),
            min(item["source_ranks"].values()),
            candidate_identity(item),
        )
    )
    return output
