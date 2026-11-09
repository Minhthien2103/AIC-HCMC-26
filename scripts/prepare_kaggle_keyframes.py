#!/usr/bin/env python3
"""Build a zero-copy canonical keyframe tree from attached Kaggle inputs.

The preprocessing notebooks store images under ``Lxx_Vxxx/keyframes`` while
the online pipeline expects ``KEYFRAMES_DIR/Lxx_Vxxx/<frame_name>``. This
script discovers every attached source, validates all names against canonical
Zilliz metadata, and links video directories instead of copying image bytes.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import polars as pl


VIDEO_RE = re.compile(r"^L\d{2}_V\d{3}$", re.IGNORECASE)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
GENERATED_MARKER = ".aic_generated_keyframe_overlay"


@dataclass(frozen=True)
class SourceDirectory:
    video_id: str
    path: Path
    names: frozenset[str]


def _image_names(path: Path) -> frozenset[str]:
    names: set[str] = set()
    with os.scandir(path) as entries:
        for entry in entries:
            if entry.is_file(follow_symlinks=True) and Path(entry.name).suffix.lower() in IMAGE_SUFFIXES:
                names.add(entry.name)
    return frozenset(names)


def discover_keyframe_directories(input_root: str | Path) -> list[SourceDirectory]:
    """Find supported layouts while pruning image directories from traversal."""
    root = Path(input_root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Kaggle input root does not exist: {root}")

    discovered: dict[tuple[str, str], SourceDirectory] = {}
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        directory = Path(current)
        video_id: str | None = None
        source: Path | None = None

        if directory.name.lower() == "keyframes" and VIDEO_RE.fullmatch(directory.parent.name):
            video_id, source = directory.parent.name.upper(), directory
        elif VIDEO_RE.fullmatch(directory.name) and directory.parent.name.lower() == "keyframes":
            video_id, source = directory.name.upper(), directory
        elif VIDEO_RE.fullmatch(directory.name) and "keyframes" in {
            name.lower() for name in directory_names
        }:
            child_name = next(name for name in directory_names if name.lower() == "keyframes")
            video_id, source = directory.name.upper(), directory / child_name
            directory_names.remove(child_name)
        elif VIDEO_RE.fullmatch(directory.name) and any(
            Path(name).suffix.lower() in IMAGE_SUFFIXES for name in file_names
        ):
            video_id, source = directory.name.upper(), directory

        if source is not None and video_id is not None:
            resolved = source.resolve()
            key = (video_id, str(resolved))
            discovered[key] = SourceDirectory(video_id, resolved, _image_names(resolved))
            if source == directory:
                directory_names.clear()

    return sorted(discovered.values(), key=lambda item: (item.video_id, str(item.path)))


def required_keyframes(metadata_path: str | Path) -> dict[str, frozenset[str]]:
    """Read exact filenames from canonical metadata without inventing aliases."""
    frame = pl.read_parquet(Path(metadata_path).expanduser())
    required_columns = {"video_id", "frame_name"}
    missing = required_columns - set(frame.columns)
    if missing:
        raise RuntimeError(f"Canonical metadata is missing columns: {sorted(missing)}")

    requirements: dict[str, set[str]] = defaultdict(set)
    for row in frame.select("video_id", "frame_name").iter_rows(named=True):
        value = str(row["frame_name"])
        name = value if Path(value).suffix else f"{value}.jpg"
        requirements[str(row["video_id"]).upper()].add(name)
    return {video_id: frozenset(names) for video_id, names in requirements.items()}


def _replace_directory_link(target: Path, source: Path) -> None:
    if target.is_symlink():
        if target.resolve() == source.resolve():
            return
        target.unlink()
    elif target.exists():
        raise RuntimeError(
            f"Refusing to replace a real directory in generated output: {target}"
        )
    target.symlink_to(source, target_is_directory=True)


def _prepare_overlay(
    target: Path,
    required_names: frozenset[str],
    candidates: list[SourceDirectory],
) -> None:
    if target.is_symlink():
        target.unlink()
    if target.exists() and not (target / GENERATED_MARKER).is_file():
        raise RuntimeError(
            f"Refusing to modify an unmarked real directory in generated output: {target}"
        )
    target.mkdir(parents=True, exist_ok=True)
    (target / GENERATED_MARKER).touch(exist_ok=True)

    by_name: dict[str, Path] = {}
    for candidate in candidates:
        for name in sorted(required_names & candidate.names):
            by_name.setdefault(name, candidate.path / name)
    for name in sorted(required_names):
        destination = target / name
        source = by_name[name]
        if destination.is_symlink():
            if destination.resolve() == source.resolve():
                continue
            destination.unlink()
        elif destination.exists():
            raise RuntimeError(f"Refusing to replace generated overlay entry: {destination}")
        destination.symlink_to(source)


def prepare_keyframe_tree(
    *,
    input_root: str | Path,
    metadata_path: str | Path,
    output_root: str | Path,
    manifest_path: str | Path | None = None,
) -> dict:
    """Validate full canonical coverage and create directory/overlay symlinks."""
    sources = discover_keyframe_directories(input_root)
    if not sources:
        raise RuntimeError(
            f"No Lxx_Vxxx/keyframes directories were found below {input_root}"
        )
    requirements = required_keyframes(metadata_path)
    by_video: dict[str, list[SourceDirectory]] = defaultdict(list)
    for source in sources:
        by_video[source.video_id].append(source)

    output = Path(output_root).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    selections: list[dict] = []
    errors: list[str] = []

    for video_id, required_names in sorted(requirements.items()):
        candidates = by_video.get(video_id, [])
        if not candidates:
            errors.append(f"{video_id}: no attached keyframe directory")
            continue

        complete = [candidate for candidate in candidates if required_names <= candidate.names]
        target = output / video_id
        if complete:
            selected = min(
                complete,
                key=lambda candidate: (len(candidate.names - required_names), str(candidate.path)),
            )
            _replace_directory_link(target, selected.path)
            selections.append(
                {
                    "video_id": video_id,
                    "mode": "directory_symlink",
                    "source": str(selected.path),
                    "required_frames": len(required_names),
                    "candidate_count": len(candidates),
                }
            )
            continue

        union_names = frozenset().union(*(candidate.names for candidate in candidates))
        missing = sorted(required_names - union_names)
        if missing:
            preview = ", ".join(missing[:5])
            errors.append(
                f"{video_id}: missing {len(missing)}/{len(required_names)} exact frames; {preview}"
            )
            continue

        _prepare_overlay(target, required_names, candidates)
        selections.append(
            {
                "video_id": video_id,
                "mode": "file_symlink_overlay",
                "sources": [str(candidate.path) for candidate in candidates],
                "required_frames": len(required_names),
                "candidate_count": len(candidates),
            }
        )

    if errors:
        preview = "\n".join(f"  - {message}" for message in errors[:20])
        raise FileNotFoundError(
            f"Attached Kaggle keyframes do not cover canonical Zilliz metadata "
            f"({len(errors)} videos failed):\n{preview}"
        )

    manifest = {
        "schema_version": "aic_kaggle_keyframe_tree_v1",
        "input_root": str(Path(input_root).expanduser()),
        "metadata_path": str(Path(metadata_path).expanduser()),
        "output_root": str(output),
        "discovered_directories": len(sources),
        "canonical_videos": len(requirements),
        "canonical_frames": sum(len(names) for names in requirements.values()),
        "selections": selections,
    }
    if manifest_path is not None:
        destination = Path(manifest_path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--metadata-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = prepare_keyframe_tree(
        input_root=args.input_root,
        metadata_path=args.metadata_path,
        output_root=args.output_root,
        manifest_path=args.manifest,
    )
    overlays = sum(item["mode"] == "file_symlink_overlay" for item in manifest["selections"])
    print(
        f"Keyframes: {manifest['canonical_frames']:,} exact files across "
        f"{manifest['canonical_videos']} videos"
    )
    print(
        f"Sources: {manifest['discovered_directories']} directories; "
        f"directory links={manifest['canonical_videos'] - overlays}, overlays={overlays}"
    )
    print(f"Canonical keyframe root: {manifest['output_root']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
