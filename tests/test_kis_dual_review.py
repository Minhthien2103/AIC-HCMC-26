from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.online_pipeline.rank_fusion import candidate_identity, fuse_rankings
from src.online_pipeline.text_evidence import OfflineEvidenceCache
from src.submission.query_parser import load_query_specs
from src.submission.review import apply_review


def _candidate(index: int) -> dict:
    return {
        "faiss_idx": index,
        "video_id": f"L21_V{index:03d}",
        "keyframe_name": f"{index:03d}",
        "frame_id": index,
        "score": 1.0 - index / 100,
    }


def test_equal_rrf_preserves_source_provenance_and_deduplicates():
    fused = fuse_rankings(
        [
            ("clip_vitb32", [_candidate(1), _candidate(1), _candidate(2)]),
            ("clip_vith14", [_candidate(2), _candidate(3)]),
        ],
        rrf_k=60,
    )

    assert [candidate["faiss_idx"] for candidate in fused] == [2, 1, 3]
    assert fused[0]["source_ranks"] == {"clip_vitb32": 3, "clip_vith14": 1}
    assert len({candidate_identity(candidate) for candidate in fused}) == len(fused)


def test_offline_evidence_fails_fast_for_missing_or_uncovered_cache(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        OfflineEvidenceCache(tmp_path / "missing.json", required=True)

    cache_path = tmp_path / "evidence.json"
    cache_path.write_text(json.dumps({"query-a": {"documents": []}}), encoding="utf-8")
    cache = OfflineEvidenceCache(cache_path, required=True)
    assert cache.context("query-a") == ""
    with pytest.raises(KeyError, match="query-b"):
        cache.context("query-b")


def test_review_can_only_reorder_generated_top_candidates(tmp_path: Path):
    candidates = [_candidate(index) for index in range(1, 6)]
    key1, key2, key3 = (candidate_identity(candidates[index]) for index in range(3))
    manifest = tmp_path / "review.json"
    manifest.write_text(
        json.dumps({"queries": {"query-p1-1-kis": {"pin": [key2, key1], "keep": [], "reject": [key3]}}}),
        encoding="utf-8",
    )

    ordered = apply_review(candidates, "query-p1-1-kis", manifest, limit=3)

    assert [candidate_identity(candidate) for candidate in ordered] == [key2, key1, candidate_identity(candidates[3]), candidate_identity(candidates[4]), key3]

    manifest.write_text(
        json.dumps({"queries": {"query-p1-1-kis": {"pin": ["faiss:999"], "keep": [], "reject": []}}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="top-3"):
        apply_review(candidates, "query-p1-1-kis", manifest, limit=3)


def test_review_can_pin_or_reject_only_generated_videos(tmp_path: Path):
    candidates = [_candidate(index) for index in range(1, 6)]
    manifest = tmp_path / "review-video.json"
    manifest.write_text(
        json.dumps({"queries": {"query-p1-1-kis": {
            "pin": [], "keep": [], "reject": [],
            "pin_video": [candidates[1]["video_id"]],
            "keep_video": [], "reject_video": [candidates[0]["video_id"]],
        }}}),
        encoding="utf-8",
    )
    ordered = apply_review(candidates, "query-p1-1-kis", manifest, limit=3)
    assert ordered[0]["video_id"] == candidates[1]["video_id"]
    assert ordered[-1]["video_id"] == candidates[0]["video_id"]


def test_btc_labelled_trake_queries_keep_four_events_in_source_order(tmp_path: Path):
    for query_id in (4, 16):
        (tmp_path / f"query-p1-{query_id}-trake.txt").write_text(
            "Video description.\nE1: first.\nE2: second.\nE3: third.\nE4: fourth.\n",
            encoding="utf-8",
        )
    (tmp_path / "query-p1-18-trake.txt").write_text(
        "Video description.\nE1: first.\nE2: second.\nE2: third.\nE4: fourth.\n",
        encoding="utf-8",
    )
    with pytest.warns(RuntimeWarning, match="labels"):
        specs = load_query_specs(queries_dir=tmp_path)
    trake = {spec.query_id: spec for spec in specs if spec.query_type == "trake"}

    assert set(trake) == {"query-p1-4-trake", "query-p1-16-trake", "query-p1-18-trake"}
    assert all(len(spec.events) == 4 for spec in trake.values())
    assert trake["query-p1-18-trake"].events[1] != trake["query-p1-18-trake"].events[2]


def test_trake_event_labels_without_colons_are_parsed(tmp_path: Path):
    query = tmp_path / "query-p1-16-trake.txt"
    query.write_text(
        "Mở đầu là cận cảnh đầu một con lân.\n"
        "E1 Hai con rồng vàng xoay vòng.\n"
        "E2 Con lân hoàn tất cú xoay người trên trụ.\n"
        "E3 Dùi chạm vào kẻng đồng múa lân.\n",
        encoding="utf-8",
    )

    spec = load_query_specs(queries_dir=tmp_path)[0]

    assert spec.description == "Mở đầu là cận cảnh đầu một con lân."
    assert spec.events == (
        "Hai con rồng vàng xoay vòng",
        "Con lân hoàn tất cú xoay người trên trụ",
        "Dùi chạm vào kẻng đồng múa lân",
    )
