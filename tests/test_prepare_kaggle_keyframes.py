from pathlib import Path

import polars as pl
import pytest

from scripts.prepare_kaggle_keyframes import prepare_keyframe_tree


def _metadata(path: Path, rows: list[dict]) -> None:
    pl.from_dicts(rows).write_parquet(path)


def test_prepare_keyframes_links_complete_video_directory(tmp_path: Path):
    source = tmp_path / "input" / "dataset-a" / "L21_V001" / "keyframes"
    source.mkdir(parents=True)
    (source / "frame_000008.jpg").touch()
    (source / "frame_000020.jpg").touch()
    metadata = tmp_path / "metadata.parquet"
    _metadata(
        metadata,
        [
            {"video_id": "L21_V001", "frame_name": "frame_000008.jpg"},
            {"video_id": "L21_V001", "frame_name": "frame_000020.jpg"},
        ],
    )

    manifest = prepare_keyframe_tree(
        input_root=tmp_path / "input",
        metadata_path=metadata,
        output_root=tmp_path / "working" / "keyframes",
        manifest_path=tmp_path / "working" / "manifest.json",
    )

    target = tmp_path / "working" / "keyframes" / "L21_V001"
    assert target.is_symlink()
    assert target.resolve() == source.resolve()
    assert manifest["canonical_frames"] == 2
    assert manifest["selections"][0]["mode"] == "directory_symlink"


def test_prepare_keyframes_builds_overlay_for_split_video(tmp_path: Path):
    first = tmp_path / "input" / "dataset-a" / "L26_V005" / "keyframes"
    second = tmp_path / "input" / "dataset-b" / "L26_V005" / "keyframes"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "frame_000001.jpg").touch()
    (second / "frame_000002.jpg").touch()
    metadata = tmp_path / "metadata.parquet"
    _metadata(
        metadata,
        [
            {"video_id": "L26_V005", "frame_name": "frame_000001.jpg"},
            {"video_id": "L26_V005", "frame_name": "frame_000002.jpg"},
        ],
    )

    manifest = prepare_keyframe_tree(
        input_root=tmp_path / "input",
        metadata_path=metadata,
        output_root=tmp_path / "working" / "keyframes",
    )

    target = tmp_path / "working" / "keyframes" / "L26_V005"
    assert target.is_dir() and not target.is_symlink()
    assert (target / "frame_000001.jpg").is_symlink()
    assert (target / "frame_000002.jpg").is_symlink()
    assert manifest["selections"][0]["mode"] == "file_symlink_overlay"


def test_prepare_keyframes_fails_on_exact_name_mismatch(tmp_path: Path):
    source = tmp_path / "input" / "dataset-a" / "L21_V001" / "keyframes"
    source.mkdir(parents=True)
    (source / "008.jpg").touch()
    metadata = tmp_path / "metadata.parquet"
    _metadata(metadata, [{"video_id": "L21_V001", "frame_name": "frame_000008.jpg"}])

    with pytest.raises(FileNotFoundError, match="frame_000008.jpg"):
        prepare_keyframe_tree(
            input_root=tmp_path / "input",
            metadata_path=metadata,
            output_root=tmp_path / "working" / "keyframes",
        )
