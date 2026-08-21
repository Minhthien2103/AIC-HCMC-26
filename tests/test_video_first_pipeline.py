from __future__ import annotations

import polars as pl

from src.online_pipeline.frame_neighborhood import FrameNeighborhood
from src.online_pipeline.video_evidence import d_hondt_allocate, fuse_video_rankings, stratified_candidates


def _frame(video: str, frame: int, index: int, score: float = 1.0) -> dict:
    return {"video_id": video, "frame_id": frame, "keyframe_name": f"{frame:03d}", "faiss_idx": index, "score": score}


def test_video_rrf_collapses_frames_and_preserves_source_provenance():
    videos = fuse_video_rankings([
        ("clip", [_frame("V1", 10, 0), _frame("V1", 20, 1), _frame("V2", 30, 2)]),
        ("metadata", [{"video_id": "V2", "score": 0.8}, {"video_id": "V1", "score": 0.7}]),
    ])
    assert [item["video_id"] for item in videos] == ["V1", "V2"]
    assert videos[0]["video_source_ranks"] == {"clip": 1, "metadata": 2}
    assert videos[0]["video_source_frame_ranks"]["clip"] == 1


def test_strata_and_dhondt_preserve_video_coverage_and_unique_frames():
    videos = [
        {"video_id": "A", "rrf_score": 0.5, "video_rank": 1, "video_source_ranks": {}, "video_source_frame_ranks": {}},
        {"video_id": "B", "rrf_score": 0.25, "video_rank": 2, "video_source_ranks": {}, "video_source_frame_ranks": {}},
    ]
    grouped = {"A": [_frame("A", 1, 1), _frame("A", 2, 2), _frame("A", 3, 3)], "B": [_frame("B", 4, 4), _frame("B", 5, 5)]}
    assert [item["video_id"] for item in stratified_candidates(grouped, videos, limit=4)] == ["A", "B", "A", "B"]
    allocated = d_hondt_allocate(videos, grouped, limit=5)
    assert {item["video_id"] for item in allocated} == {"A", "B"}
    assert len({(item["video_id"], item["frame_id"]) for item in allocated}) == len(allocated)


def test_frame_neighborhood_is_anchor_first_deterministic_and_bounded():
    metadata = pl.DataFrame({
        "video_id": ["V", "V", "V"],
        "faiss_idx": [0, 1, 2],
        "frame_id": [100, 200, 400],
    })
    neighbourhood = FrameNeighborhood(metadata)
    proposals = neighbourhood.proposals(_frame("V", 200, 1), count=7)
    assert proposals[0]["frame_id"] == 200
    assert all(150 <= item["frame_id"] <= 300 for item in proposals)
    assert len({item["frame_id"] for item in proposals}) == len(proposals)
    assert neighbourhood.proposals(_frame("V", 200, 1), count=7) == proposals


def test_frame_neighborhood_sequence_honours_only_given_edges():
    metadata = pl.DataFrame({"video_id": ["V"] * 3, "faiss_idx": [0, 1, 2], "frame_id": [100, 200, 300]})
    neighbourhood = FrameNeighborhood(metadata)
    sequence = [_frame("V", 200, 1), _frame("V", 100, 0)]
    assert neighbourhood.sequence_proposals(sequence, edges=[], limit=1)[0][0]["frame_id"] == 200
    assert neighbourhood.sequence_proposals(sequence, edges=[(0, 1)], limit=10) == []
