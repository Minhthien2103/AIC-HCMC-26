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
    temporal_edges: list[tuple[int, int]] | None = None,
    beam_size: int = 25,
    max_sequences: int = 100,
) -> list[dict[str, Any]]:
    """Return complete event chains satisfying explicit temporal edges only.

    Each event list is kept separate, so a missing event makes alignment
    impossible rather than silently shortening the returned sequence.
    """
    if not event_lists or any(not candidates for candidates in event_lists):
        return []

    edges = list(temporal_edges or [])
    event_count = len(event_lists)
    if any(not (0 <= before < event_count and 0 <= after < event_count and before != after) for before, after in edges):
        raise ValueError("Temporal edge indexes must reference distinct events")

    def _valid_partial(events: list[dict[str, Any]]) -> bool:
        for before, after in edges:
            if before < len(events) and after < len(events):
                if int(events[before]["frame_id"]) >= int(events[after]["frame_id"]):
                    return False
        return True

    states = [_State(score=0.0, events=[])]
    for candidates in event_lists:
        next_states: list[_State] = []
        for state in states:
            for candidate in candidates:
                events = [*state.events, candidate]
                if not _valid_partial(events):
                    continue
                next_states.append(_State(
                    score=state.score + float(candidate.get("score", 0.0)),
                    events=events,
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
