"""Human review artefacts for KIS candidates without allowing new guesses."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from src.online_pipeline.rank_fusion import candidate_identity


REVIEW_ACTIONS = ("pin", "keep", "reject")
VIDEO_REVIEW_ACTIONS = ("pin_video", "keep_video", "reject_video")


def _json_default(value: Any) -> str:
    return str(value)


def _candidate_payload(candidate: dict[str, Any], rank: int) -> dict[str, Any]:
    payload = candidate.copy()
    payload["candidate_key"] = candidate_identity(candidate)
    payload["fusion_rank"] = rank
    return payload


def write_review_assets(
    query_id: str,
    candidates: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    image_path_for,
    limit: int = 20,
) -> Path:
    """Write pipeline-only candidate provenance plus a five-column sheet.

    This works for KIS and QA frames.  TRAKE sequences use the first event as
    their representative tile; full event provenance remains in JSON.
    """
    if limit < 1:
        raise ValueError("review limit must be positive")
    query_dir = Path(output_dir) / query_id
    query_dir.mkdir(parents=True, exist_ok=True)
    selected = candidates[:limit]
    payload = [_candidate_payload(candidate, rank) for rank, candidate in enumerate(selected, start=1)]
    (query_dir / "candidates.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8"
    )

    width, height, columns = 260, 190, 5
    rows = max(1, (len(selected) + columns - 1) // columns)
    sheet = Image.new("RGB", (columns * width, rows * height), "white")
    draw = ImageDraw.Draw(sheet)
    for position, candidate in enumerate(selected):
        x, y = (position % columns) * width, (position // columns) * height
        display = candidate.get("events", [candidate])[0] if candidate.get("events") else candidate
        try:
            with Image.open(image_path_for(display)) as opened:
                image = opened.convert("RGB")
                image.thumbnail((width - 8, height - 46))
                sheet.paste(image, (x + (width - image.width) // 2, y + 4))
        except Exception as exc:
            draw.text((x + 8, y + 35), f"Image unavailable\n{exc}", fill="red")
        label = (
            f"#{position + 1} {candidate.get('video_id')} / {display.get('frame_id', display.get('keyframe_name'))}\n"
            f"sources: {', '.join((candidate.get('source_ranks') or {}).keys())}"
        )
        draw.text((x + 4, y + height - 38), label, fill="black")
    sheet.save(query_dir / "contact_sheet.jpg", quality=92)

    template_path = query_dir / "review.json"
    if not template_path.exists():
        template_path.write_text(
            json.dumps(
                {
                    "query_id": query_id,
                    "allowed_candidate_keys": [candidate_identity(item) for item in selected],
                    "allowed_video_ids": list(dict.fromkeys(str(item.get("video_id", "")) for item in selected if item.get("video_id"))),
                    "pin": [],
                    "keep": [],
                    "reject": [],
                    "pin_video": [],
                    "keep_video": [],
                    "reject_video": [],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return query_dir


def write_review_manifest_template(output_dir: str | Path, query_ids: list[str]) -> Path:
    """Collect per-query top-20 keys into one editable final-review manifest."""
    root = Path(output_dir)
    queries: dict[str, dict[str, list[str]]] = {}
    for query_id in query_ids:
        candidates_path = root / query_id / "candidates.json"
        if not candidates_path.exists():
            continue
        candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
        if not isinstance(candidates, list):
            raise ValueError(f"Invalid generated review candidates: {candidates_path}")
        queries[query_id] = {
            "pin": [],
            "keep": [],
            "reject": [],
            "allowed_candidate_keys": [str(item["candidate_key"]) for item in candidates],
            "allowed_video_ids": list(dict.fromkeys(str(item.get("video_id", "")) for item in candidates if item.get("video_id"))),
            "pin_video": [],
            "keep_video": [],
            "reject_video": [],
        }
    path = root / "review_manifest.template.json"
    path.write_text(json.dumps({"queries": queries}, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _review_for_query(manifest_path: str | Path, query_id: str) -> dict[str, list[str]]:
    path = Path(manifest_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    queries = payload.get("queries", payload) if isinstance(payload, dict) else {}
    item = queries.get(query_id, {}) if isinstance(queries, dict) else {}
    if not isinstance(item, dict):
        raise ValueError(f"Review entry for {query_id} must be an object")
    result: dict[str, list[str]] = {}
    for action in (*REVIEW_ACTIONS, *VIDEO_REVIEW_ACTIONS):
        values = item.get(action, [])
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            raise ValueError(f"Review action {action} for {query_id} must be a list of generated identities")
        result[action] = values
    return result


def apply_review(
    candidates: list[dict[str, Any]],
    query_id: str,
    manifest_path: str | Path | None,
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Apply pin/keep/reject without accepting candidate identities not found.

    ``keep`` is intentionally an audit action, not an arbitrary score boost:
    after explicitly ordered pins, every non-rejected pipeline candidate keeps
    its fusion order.  Rejected candidates remain in the output tail so the
    final CSV can still contain up to 100 rows.
    """
    if manifest_path is None:
        return candidates
    review = _review_for_query(manifest_path, query_id)
    allowed = {candidate_identity(candidate) for candidate in candidates[:limit]}
    allowed_videos = {str(candidate.get("video_id", "")).removesuffix(".mp4") for candidate in candidates[:limit]}
    actions: dict[str, str] = {}
    for action in REVIEW_ACTIONS:
        for key in review[action]:
            if key not in allowed:
                raise ValueError(
                    f"Review {query_id} references {key}, which is not among the generated top-{limit} candidates"
                )
            if key in actions:
                raise ValueError(f"Review {query_id} assigns {key} to both {actions[key]} and {action}")
            actions[key] = action

    video_actions: dict[str, str] = {}
    for action in VIDEO_REVIEW_ACTIONS:
        for video_id in review[action]:
            normalized = str(video_id).removesuffix(".mp4")
            if normalized not in allowed_videos:
                raise ValueError(f"Review {query_id} references video {video_id}, which is not among the generated top-{limit} candidates")
            if normalized in video_actions:
                raise ValueError(f"Review {query_id} assigns video {normalized} to both {video_actions[normalized]} and {action}")
            video_actions[normalized] = action

    by_key = {candidate_identity(candidate): candidate for candidate in candidates}
    for key, action in actions.items():
        if action in {"pin", "keep"}:
            video_id = str(by_key[key].get("video_id", "")).removesuffix(".mp4")
            if video_actions.get(video_id) == "reject_video":
                raise ValueError(f"Review {query_id} keeps/pins {key} but rejects its video {video_id}")
    pinned = [by_key[key] for key in review["pin"]]
    pinned_keys = set(review["pin"])
    video_pinned = [
        candidate for video_id in review["pin_video"]
        for candidate in candidates
        if str(candidate.get("video_id", "")).removesuffix(".mp4") == video_id and candidate_identity(candidate) not in pinned_keys
        and actions.get(candidate_identity(candidate)) != "reject"
    ]
    video_pinned_keys = {candidate_identity(candidate) for candidate in video_pinned}
    non_rejected = [
        candidate for candidate in candidates
        if candidate_identity(candidate) not in pinned_keys
        and candidate_identity(candidate) not in video_pinned_keys
        and actions.get(candidate_identity(candidate)) != "reject"
        and video_actions.get(str(candidate.get("video_id", "")).removesuffix(".mp4")) != "reject_video"
    ]
    rejected = [
        candidate for candidate in candidates
        if candidate_identity(candidate) not in pinned_keys
        and candidate_identity(candidate) not in video_pinned_keys
        and (actions.get(candidate_identity(candidate)) == "reject"
        or video_actions.get(str(candidate.get("video_id", "")).removesuffix(".mp4")) == "reject_video"
        )
    ]
    return [*pinned, *video_pinned, *non_rejected, *rejected]
