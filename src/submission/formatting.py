"""Competition output normalization shared by CLI, UI and validator."""

from __future__ import annotations

import re
from typing import Any


ERROR_MARKERS = (
    "[vlm error",
    "failed to run vlm",
    "error during vlm",
    "error during inference",
    "manual review required",
    "bitsandbytes",
    "probability tensor contains",
)


def normalize_video_id(value: Any) -> str:
    video_id = str(value or "").strip()
    if video_id.lower().endswith(".mp4"):
        video_id = video_id[:-4]
    return video_id


def normalize_frame_id(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("frame_id must be an integer, not boolean")
    if isinstance(value, int):
        return value
    text = str(value or "").strip()
    if not re.fullmatch(r"[+-]?\d+", text):
        raise ValueError(f"frame_id is not an integer: {value!r}")
    return int(text)


def _strip_markdown(text: str) -> str:
    # Remove common model wrappers, including malformed nested image wrappers
    # seen in failed Qwen generations, before applying the length contract.
    text = re.sub(r"!\[[^\]]*\]\([^\n]*?\)", "", text)
    text = re.sub(r"(?:!\[\]\(){2,}", "", text)
    text = re.sub(r"^\s*(?:answer|response|final answer)\s*:\s*", "", text, flags=re.IGNORECASE)
    text = text.replace("```text", "").replace("```", "")
    return text.strip()


def normalize_answer(value: Any, max_chars: int = 100) -> str:
    raw = str(value or "")
    if "![](" in raw or "![" in raw:
        raise ValueError("answer contains a Markdown/image wrapper")
    answer = _strip_markdown(raw)
    answer = re.sub(r"\s+", " ", answer).strip()
    if not answer:
        raise ValueError("answer is empty")
    lower = answer.lower()
    if any(marker in lower for marker in ERROR_MARKERS):
        raise ValueError(f"answer contains an error marker: {answer[:80]!r}")
    if len(answer) > max_chars:
        truncated = answer[:max_chars].rsplit(" ", 1)[0].rstrip(" ,.;:!?-")
        answer = truncated or answer[:max_chars]
    if len(answer) > max_chars:
        answer = answer[:max_chars]
    return answer


def format_kis_row(result: dict[str, Any]) -> list[Any]:
    return [normalize_video_id(result["video_id"]), normalize_frame_id(result["frame_id"])]


def format_qa_row(result: dict[str, Any], max_chars: int = 100) -> list[Any]:
    return [
        normalize_video_id(result["video_id"]),
        normalize_frame_id(result["frame_id"]),
        normalize_answer(result.get("answer"), max_chars=max_chars),
    ]


def format_trake_row(result: dict[str, Any]) -> list[Any]:
    events = result.get("events") or []
    if not result.get("is_valid_sequence") or not events:
        raise ValueError("TRAKE result is not a complete valid sequence")
    frame_ids = [normalize_frame_id(event["frame_id"]) for event in events]
    if any(left >= right for left, right in zip(frame_ids, frame_ids[1:])):
        raise ValueError("TRAKE frame IDs must increase strictly")
    return [normalize_video_id(result["video_id"]), *frame_ids]
