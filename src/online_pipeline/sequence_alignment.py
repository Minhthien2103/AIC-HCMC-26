"""Temporal dynamic-programming/beam alignment for event retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class _State:
    score: float
    events: list[dict[str, Any]]


def align_event_candidates(
    event_lists: list[list[dict[str, Any]]],
    *,
    beam_size: int = 25,
    max_sequences: int = 100,
) -> list[dict[str, Any]]:
    """Return complete strictly chronological event chains.

    Each event list is kept separate, so a missing event makes alignment
    impossible rather than silently shortening the returned sequence.
    """
    if not event_lists or any(not candidates for candidates in event_lists):
        return []

    states = [_State(score=0.0, events=[])]
    for candidates in event_lists:
        next_states: list[_State] = []
        for state in states:
            previous_frame = state.events[-1]["frame_id"] if state.events else None
            for candidate in candidates:
                frame_id = int(candidate["frame_id"])
                if previous_frame is not None and frame_id <= int(previous_frame):
                    continue
                
                # Apply a small penalty for very large temporal gaps between events
                # Assumes 25 fps: 1 minute = 1500 frames
                penalty = 0.0
                if previous_frame is not None:
                    gap = frame_id - int(previous_frame)
                    penalty = min(gap * 0.00001, 0.05)  # Cap penalty at 0.05
                
                next_states.append(_State(
                    score=state.score + float(candidate.get("score", 0.0)) - penalty,
                    events=[*state.events, candidate],
                ))
        if not next_states:
            return []
        next_states.sort(key=lambda state: state.score, reverse=True)
        deduped: list[_State] = []
        seen: set[tuple[int, ...]] = set()
        for state in next_states:
            key = tuple(int(event["frame_id"]) for event in state.events)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(state)
            if len(deduped) >= beam_size:
                break
        states = deduped

    return [
        {"events": state.events, "sequence_score": state.score, "is_valid_sequence": True}
        for state in states[:max_sequences]
    ]
