"""Deterministic MP4-frame proposals around supplied keyframes.

The organisers score original MP4 frame ids, not only extracted keyframe ids.
Map-keyframes tells us the local interval represented by a keyframe, so a
small deterministic neighbourhood improves localisation without decoding raw
video or guessing arbitrary timestamps.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from typing import Any

import numpy as np


class FrameNeighborhood:
    def __init__(self, metadata):
        """Build immutable per-video frame-id lookup from metadata.parquet."""
        rows = metadata.select(["video_id", "faiss_idx", "frame_id"]).to_dicts()
        by_video: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._by_faiss: dict[int, tuple[str, int, int]] = {}
        for row in rows:
            video_id = str(row["video_id"])
            frame_id = int(row["frame_id"])
            faiss_idx = int(row["faiss_idx"])
            by_video[video_id].append((frame_id, faiss_idx))
        self._video_frames: dict[str, list[int]] = {}
        for video_id, values in by_video.items():
            values.sort()
            frames = [frame for frame, _index in values]
            self._video_frames[video_id] = frames
            for position, (frame, faiss_idx) in enumerate(values):
                self._by_faiss[faiss_idx] = (video_id, position, frame)

    def _location(self, candidate: dict[str, Any]) -> tuple[str, int, int]:
        faiss_idx = candidate.get("faiss_idx")
        if faiss_idx is not None and int(faiss_idx) in self._by_faiss:
            return self._by_faiss[int(faiss_idx)]
        video_id = str(candidate["video_id"])
        anchor = int(candidate["frame_id"])
        frames = self._video_frames.get(video_id)
        if not frames:
            raise KeyError(f"No map-keyframe metadata for {video_id}")
        position = bisect_left(frames, anchor)
        if position == len(frames) or (position and abs(frames[position - 1] - anchor) <= abs(frames[position] - anchor)):
            position -= 1
        return video_id, max(0, position), frames[max(0, position)]

    def proposals(self, candidate: dict[str, Any], *, count: int = 7) -> list[dict[str, Any]]:
        """Return anchor-first, evenly spread integer ids in its local cell."""
        if count < 1:
            return []
        video_id, position, anchor = self._location(candidate)
        frames = self._video_frames[video_id]
        previous = frames[position - 1] if position else 0
        following = frames[position + 1] if position + 1 < len(frames) else frames[position]
        left = (previous + anchor) // 2 if position else max(0, anchor - max(1, (following - anchor) // 2))
        right = (anchor + following) // 2 if position + 1 < len(frames) else anchor + max(1, (anchor - previous) // 2)
        samples = {int(value) for value in np.linspace(left, right, num=max(2, count), dtype=np.int64)}
        samples.add(anchor)
        ordered = [anchor, *sorted((value for value in samples if value != anchor), key=lambda value: (abs(value - anchor), value))]
        result: list[dict[str, Any]] = []
        for rank, frame_id in enumerate(ordered[:count], start=1):
            item = candidate.copy()
            item["video_id"] = video_id
            item["frame_id"] = int(frame_id)
            item["anchor_frame_id"] = int(anchor)
            item["neighborhood_start"] = int(left)
            item["neighborhood_end"] = int(right)
            item["neighborhood_rank"] = rank
            result.append(item)
        return result

    def expand_ranked(self, candidates: list[dict[str, Any]], *, count: int = 7) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        seen: set[tuple[str, int]] = set()
        for candidate in candidates:
            for proposal in self.proposals(candidate, count=count):
                key = (str(proposal["video_id"]), int(proposal["frame_id"]))
                if key not in seen:
                    seen.add(key)
                    output.append(proposal)
        return output

    def sequence_proposals(
        self,
        sequence: list[dict[str, Any]],
        *,
        edges: list[tuple[int, int]],
        limit: int,
        count: int = 7,
    ) -> list[list[dict[str, Any]]]:
        """Generate deterministic sequence hedges, respecting known edges only."""
        if limit < 1 or not sequence:
            return []
        alternatives = [self.proposals(event, count=count) for event in sequence]
        output: list[list[dict[str, Any]]] = []
        seen: set[tuple[int, ...]] = set()

        def valid(events: list[dict[str, Any]]) -> bool:
            return all(int(events[left]["frame_id"]) < int(events[right]["frame_id"]) for left, right in edges)

        def append(events: list[dict[str, Any]]) -> None:
            key = tuple(int(event["frame_id"]) for event in events)
            if key not in seen and valid(events):
                seen.add(key)
                output.append(events)

        append([choices[0].copy() for choices in alternatives])
        # Single-event changes preserve all other high-confidence anchors.
        for event_index, choices in enumerate(alternatives):
            for choice in choices[1:]:
                proposal = [values[0].copy() for values in alternatives]
                proposal[event_index] = choice.copy()
                append(proposal)
                if len(output) >= limit:
                    return output
        # A Latin-hypercube-style sweep covers simultaneous uncertainty without
        # an exponential Cartesian product.
        sweep = 1
        while len(output) < limit:
            proposal = [choices[sweep % len(choices)].copy() for choices in alternatives]
            before = len(output)
            append(proposal)
            if len(output) == before and sweep > count * max(2, len(sequence)):
                break
            sweep += 1
        return output[:limit]
