"""Flexible query-pack parsing for the AIC qualification format.

The official query packs have changed their presentation between rounds.  The
runner therefore accepts either a manifest or the filename conventions used by
the organizers, while keeping one normalized object for every query.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


QUERY_TYPES = {"kis", "qa", "trake"}
_TYPE_RE = re.compile(r"(?:^|[_-])(kis|qa|trake)(?:\.[^.]+)?$", re.IGNORECASE)


@dataclass(frozen=True)
class QuerySpec:
    query_id: str
    query_type: str
    description: str = ""
    question: str = ""
    events: tuple[str, ...] = field(default_factory=tuple)
    source_path: str = ""

    def __post_init__(self) -> None:
        qtype = self.query_type.lower().strip()
        if qtype not in QUERY_TYPES:
            raise ValueError(f"Unsupported query type: {self.query_type!r}")
        object.__setattr__(self, "query_type", qtype)
        if not self.query_id.strip():
            raise ValueError("Query id cannot be empty")


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _query_type_from_name(name: str) -> str | None:
    stem = Path(name).stem
    match = _TYPE_RE.search(stem)
    return match.group(1).lower() if match else None


def _strip_label(text: str, label: str) -> str | None:
    match = re.match(rf"^\s*{re.escape(label)}\s*:\s*(.*)$", text, flags=re.IGNORECASE)
    return match.group(1).strip() if match else None


def _blocks(text: str) -> list[str]:
    return [block.strip() for block in re.split(r"\n\s*\n+", text.replace("\r\n", "\n")) if block.strip()]


def _parse_text(path: Path, query_type: str) -> QuerySpec:
    text = path.read_text(encoding="utf-8-sig").strip()
    lines = text.replace("\r\n", "\n").splitlines()
    description = ""
    question = ""
    events: list[str] = []

    if query_type == "kis":
        description = text
    elif query_type == "qa":
        explicit_description: list[str] = []
        explicit_question: list[str] = []
        active: list[str] | None = None
        for line in lines:
            desc_value = _strip_label(line, "Description")
            question_value = _strip_label(line, "Question")
            if desc_value is not None:
                active = explicit_description
                if desc_value:
                    active.append(desc_value)
            elif question_value is not None:
                active = explicit_question
                if question_value:
                    active.append(question_value)
            elif active is not None:
                active.append(line)

        if explicit_question:
            question = "\n".join(explicit_question).strip()
            description = "\n".join(explicit_description).strip() or question
        else:
            blocks = _blocks(text)
            if len(blocks) >= 2:
                description, question = blocks[0], blocks[1]
            else:
                description = question = text
    else:
        before_events: list[str] = []
        after_events = False
        for line in lines:
            events_value = _strip_label(line, "Events")
            if events_value is not None:
                after_events = True
                if events_value:
                    events.extend(_split_events(events_value))
                continue
            if after_events:
                value = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
                if value:
                    events.append(value)
            else:
                before_events.append(line)
        description = "\n".join(before_events).strip() or text
        if not events:
            blocks = _blocks(text)
            if len(blocks) > 1:
                description = blocks[0]
                events = [line.strip() for line in blocks[1].splitlines() if line.strip()]

    return QuerySpec(
        query_id=path.stem,
        query_type=query_type,
        description=description,
        question=question,
        events=tuple(events),
        source_path=str(path),
    )


def _split_events(value: str) -> list[str]:
    return [part.strip() for part in re.split(r"\s*(?:;|\||,)\s*", value) if part.strip()]


def _from_manifest_item(item: dict[str, Any], source: str = "") -> QuerySpec:
    query_type = _clean(item.get("type") or item.get("query_type")).lower()
    query_id = _clean(item.get("id") or item.get("query_id") or item.get("name"))
    events_value = item.get("events") or []
    if isinstance(events_value, str):
        events = _split_events(events_value)
    else:
        events = [_clean(event) for event in events_value if _clean(event)]
    return QuerySpec(
        query_id=query_id,
        query_type=query_type,
        description=_clean(item.get("description")),
        question=_clean(item.get("question")),
        events=tuple(events),
        source_path=source,
    )


def _load_manifest(path: Path) -> list[QuerySpec]:
    raw_text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError("YAML manifest requires PyYAML; use JSON or install pyyaml") from exc
        payload = yaml.safe_load(raw_text)
    else:
        payload = json.loads(raw_text)
    items = payload.get("queries", payload) if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise ValueError("Manifest must contain a list or a top-level 'queries' list")
    return [_from_manifest_item(item, str(path)) for item in items]


def load_query_specs(queries_dir: str | Path | None = None, manifest: str | Path | None = None) -> list[QuerySpec]:
    """Load and deterministically sort query specifications."""
    if manifest:
        specs = _load_manifest(Path(manifest))
    else:
        if not queries_dir:
            raise ValueError("Either queries_dir or manifest is required")
        directory = Path(queries_dir)
        if not directory.is_dir():
            raise FileNotFoundError(f"Query directory not found: {directory}")
        specs = []
        for path in sorted(directory.iterdir()):
            if not path.is_file() or path.name.startswith("."):
                continue
            query_type = _query_type_from_name(path.name)
            if query_type:
                specs.append(_parse_text(path, query_type))
        if not specs:
            raise ValueError(f"No *_kis, *_qa or *_trake query files found in {directory}")
    return sorted(specs, key=lambda spec: spec.query_id)
