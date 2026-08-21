"""Video-level evidence fusion and deterministic candidate allocation.

The official metric rewards a correct video/frame near the top of each CSV.
Frame-only retrieval can therefore waste rank mass over many videos even when
the correct video is already present.  This module keeps the evidence for each
source auditable, fuses *videos* with equal-weight RRF, and only then allocates
frame slots to the selected videos.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any


def video_identity(candidate: dict[str, Any]) -> str:
    video_id = str(candidate.get("video_id", "")).strip().removesuffix(".mp4")
    if not video_id:
        raise ValueError("Video candidate has no video_id")
    return video_id


def collapse_to_videos(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the highest-ranked frame for each video in one source list.

    The input order is authoritative.  We deliberately do not compare raw
    scores between CLIP, OCR and Qwen; they are incomparable across models.
    """
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for frame_rank, candidate in enumerate(candidates, start=1):
        video_id = video_identity(candidate)
        if video_id in seen:
            continue
        seen.add(video_id)
        item = candidate.copy()
        item["video_id"] = video_id
        item["source_frame_rank"] = frame_rank
        output.append(item)
    return output


def fuse_video_rankings(
    rankings: Iterable[tuple[str, list[dict[str, Any]]]],
    *,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """Fuse independent video rankings with equal-weight reciprocal rank."""
    if rrf_k < 0:
        raise ValueError("rrf_k must be non-negative")

    merged: dict[str, dict[str, Any]] = {}
    for source_name, source_candidates in rankings:
        name = str(source_name).strip()
        if not name:
            raise ValueError("Video fusion source name cannot be empty")
        for rank, candidate in enumerate(collapse_to_videos(source_candidates), start=1):
            video_id = video_identity(candidate)
            item = merged.setdefault(
                video_id,
                {
                    "video_id": video_id,
                    "rrf_score": 0.0,
                    "video_source_ranks": {},
                    "video_source_frame_ranks": {},
                    "video_source_scores": {},
                },
            )
            item["rrf_score"] = float(item["rrf_score"]) + 1.0 / (rrf_k + rank)
            item["video_source_ranks"][name] = rank
            item["video_source_frame_ranks"][name] = int(candidate.get("source_frame_rank", rank))
            item["video_source_scores"][name] = float(
                candidate.get("score", candidate.get("metadata_score", 0.0))
            )

    result = list(merged.values())
    for item in result:
        item["video_source_ranks"] = dict(sorted(item["video_source_ranks"].items()))
        item["video_source_frame_ranks"] = dict(sorted(item["video_source_frame_ranks"].items()))
        item["video_source_scores"] = dict(sorted(item["video_source_scores"].items()))
    result.sort(
        key=lambda item: (
            -float(item["rrf_score"]),
            min(item["video_source_ranks"].values()),
            str(item["video_id"]),
        )
    )
    for rank, item in enumerate(result, start=1):
        item["video_rank"] = rank
    return result


def stratified_candidates(
    frames_by_video: dict[str, list[dict[str, Any]]],
    videos: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Take candidates round-robin by video so Qwen/OCR cannot see one video only."""
    if limit < 1:
        return []
    queues = {
        str(video["video_id"]): list(frames_by_video.get(str(video["video_id"]), []))
        for video in videos
    }
    output: list[dict[str, Any]] = []
    offset = 0
    while len(output) < limit:
        added = False
        for video in videos:
            video_id = str(video["video_id"])
            values = queues[video_id]
            if offset < len(values):
                output.append(values[offset].copy())
                added = True
                if len(output) >= limit:
                    break
        if not added:
            break
        offset += 1
    return output


def d_hondt_allocate(
    videos: list[dict[str, Any]],
    frames_by_video: dict[str, list[dict[str, Any]]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Allocate output slots proportionally to evidence without score tuning.

    D'Hondt uses only the comparable RRF scores.  A highly corroborated video
    naturally receives more early frame slots; weaker, distinct videos remain
    represented when the evidence is uncertain.
    """
    if limit < 1:
        return []
    queues = {
        str(video["video_id"]): list(frames_by_video.get(str(video["video_id"]), []))
        for video in videos
    }
    emitted: defaultdict[str, int] = defaultdict(int)
    video_by_id = {str(item["video_id"]): item for item in videos}
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, int | str]] = set()

    while len(output) < limit:
        available = [video_id for video_id, values in queues.items() if values]
        if not available:
            break
        video_id = min(
            available,
            key=lambda key: (
                -float(video_by_id[key]["rrf_score"]) / (emitted[key] + 1),
                int(video_by_id[key]["video_rank"]),
                key,
            ),
        )
        candidate = queues[video_id].pop(0).copy()
        identity = (video_id, candidate.get("frame_id", candidate.get("keyframe_name", "")))
        if identity in seen:
            continue
        seen.add(identity)
        emitted[video_id] += 1
        evidence = video_by_id[video_id]
        candidate["video_rank"] = int(evidence["video_rank"])
        candidate["video_rrf_score"] = float(evidence["rrf_score"])
        candidate["video_source_ranks"] = dict(evidence["video_source_ranks"])
        candidate["video_source_frame_ranks"] = dict(evidence["video_source_frame_ranks"])
        candidate["video_allocation_position"] = emitted[video_id]
        output.append(candidate)
    return output
