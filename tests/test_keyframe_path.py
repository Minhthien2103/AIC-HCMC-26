from pathlib import Path

from scripts.validate_zilliz_setup import _keyframe_path
from src import config


def test_keyframe_path_prefers_exact_zilliz_filename(tmp_path: Path):
    video_dir = tmp_path / "L21_V001"
    video_dir.mkdir()
    exact = video_dir / "frame_000008.jpg"
    exact.touch()
    (video_dir / "008.jpg").touch()

    assert config.keyframe_path("L21_V001", "frame_000008", root=tmp_path) == exact


def test_keyframe_path_maps_zilliz_name_to_legacy_archive(tmp_path: Path):
    video_dir = tmp_path / "L21_V001"
    video_dir.mkdir()
    legacy = video_dir / "008.jpg"
    legacy.touch()

    assert config.keyframe_path("L21_V001", "frame_000008.jpg", root=tmp_path) == legacy
    assert _keyframe_path(tmp_path, "L21_V001", "frame_000008") == legacy


def test_keyframe_path_keeps_expected_path_when_file_is_missing(tmp_path: Path):
    expected = tmp_path / "L21_V001" / "frame_000008.jpg"

    assert config.keyframe_path("L21_V001", "frame_000008", root=tmp_path) == expected
