"""CSV contract helpers used by the generator and validator."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, TextIO

from .formatting import ERROR_MARKERS, normalize_frame_id, normalize_video_id
from .query_parser import QuerySpec


def expected_columns(spec: QuerySpec) -> int:
    if spec.query_type == "kis":
        return 2
    if spec.query_type == "qa":
        return 3
    return 1 + len(spec.events)


def validate_rows(rows: list[list[str]], spec: QuerySpec, max_rows: int = 100) -> list[str]:
    errors: list[str] = []
    if len(rows) == 0:
        errors.append("file has no rows")
    if len(rows) > max_rows:
        errors.append(f"contains {len(rows)} rows; maximum is {max_rows}")
    if rows and rows[0] and rows[0][0].strip().lower() in {"video_id", "video", "vid"}:
        errors.append("header row is not allowed")

    seen: set[tuple[str, ...]] = set()
    for row_number, row in enumerate(rows, start=1):
        prefix = f"row {row_number}: "
        if len(row) != expected_columns(spec):
            errors.append(prefix + f"expected {expected_columns(spec)} columns, got {len(row)}")
            continue
        try:
            video_id = normalize_video_id(row[0])
            if not video_id or video_id != row[0].strip() or ".mp4" in row[0].lower():
                errors.append(prefix + "video_id must be non-empty and must not include .mp4")
            frame_ids = [normalize_frame_id(value) for value in row[1:2 if spec.query_type != "trake" else None]]
            if any(frame_id < 0 for frame_id in frame_ids):
                errors.append(prefix + "frame_id must be non-negative")
        except ValueError as exc:
            errors.append(prefix + str(exc))
            continue

        if spec.query_type == "qa":
            answer = row[2].strip()
            if not answer:
                errors.append(prefix + "answer is empty")
            if len(answer) > 100:
                errors.append(prefix + "answer exceeds 100 characters")
            if any(marker in answer.lower() for marker in ERROR_MARKERS):
                errors.append(prefix + "answer contains an error marker")
        elif spec.query_type == "trake":
            try:
                trake_frames = [normalize_frame_id(value) for value in row[1:]]
                if len(trake_frames) != len(spec.events):
                    errors.append(prefix + f"expected {len(spec.events)} TRAKE frames")
                if any(left >= right for left, right in zip(trake_frames, trake_frames[1:])):
                    errors.append(prefix + "TRAKE frame IDs must increase strictly")
            except ValueError as exc:
                errors.append(prefix + str(exc))

        key = tuple(row)
        if key in seen:
            errors.append(prefix + "duplicate row")
        seen.add(key)
    return errors


def parse_csv_text(stream: TextIO) -> list[list[str]]:
    return list(csv.reader(stream))


def write_csv(path: str | Path, rows: Iterable[Iterable[object]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerows(rows)


def read_csv_file(path: str | Path) -> list[list[str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return parse_csv_text(handle)
