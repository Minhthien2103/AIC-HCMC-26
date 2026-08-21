import json
from pathlib import Path

import pytest

from src.online_pipeline.sequence_alignment import align_event_candidates
from src.online_pipeline.fusion import ScoreFuser
from src.submission.formatting import normalize_answer, normalize_frame_id
from src.submission.io import read_csv_file, validate_rows, write_csv
from src.submission.query_parser import QuerySpec, load_query_specs


def test_query_parser_supports_qa_blocks_and_trake_events(tmp_path: Path):
    (tmp_path / "query-1-qa.txt").write_text(
        "Video about a music award\n\nHow many people are on stage?\n", encoding="utf-8"
    )
    (tmp_path / "query-2-trake.txt").write_text(
        "A person crosses a room\nEvents:\n1. Starts walking\n2. Reaches the door\n", encoding="utf-8"
    )
    specs = load_query_specs(queries_dir=tmp_path)
    assert specs[0].question == "How many people are on stage?"
    assert specs[1].events == ("Starts walking", "Reaches the door")


def test_query_parser_extracts_inline_numbered_trake_events(tmp_path: Path):
    (tmp_path / "query-4-trake.txt").write_text(
        "Tìm 4 khoảnh khắc chính khi vận động viên thực hiện cú nhảy: "
        "(1) giậm nhảy, (2) bay qua xà, (3) tiếp đất, (4) đứng dậy.",
        encoding="utf-8",
    )

    specs = load_query_specs(queries_dir=tmp_path)

    assert specs[0].description == "Tìm 4 khoảnh khắc chính khi vận động viên thực hiện cú nhảy"
    assert specs[0].events == ("giậm nhảy", "bay qua xà", "tiếp đất", "đứng dậy")


def test_query_parser_manifest(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"queries": [{
        "id": "query-7-qa",
        "type": "qa",
        "description": "award ceremony",
        "question": "How many people?",
    }]}), encoding="utf-8")
    specs = load_query_specs(manifest=manifest)
    assert specs[0].query_id == "query-7-qa"
    assert specs[0].question == "How many people?"


def test_answer_is_trimmed_at_word_boundary():
    assert len(normalize_answer("one two three four five", max_chars=12)) <= 12
    with pytest.raises(ValueError):
        normalize_answer("![](![](![](")
    with pytest.raises(ValueError):
        normalize_answer("   ")


def test_validator_rejects_header_and_mp4_but_allows_unordered_trake():
    spec = QuerySpec("q-trake", "trake", description="x", events=("a", "b", "c"))
    errors = validate_rows(
        [["video_id", "frame_id_1", "frame_id_2", "frame_id_3"], ["L21_V001.mp4", "10", "9", "12"]],
        spec,
    )
    assert any("header" in error for error in errors)
    assert any(".mp4" in error for error in errors)
    assert not any("increase strictly" in error for error in errors)


def test_alignment_requires_all_events_and_only_explicit_time_order():
    event_lists = [
        [{"frame_id": 10, "score": 0.9}],
        [{"frame_id": 5, "score": 0.99}, {"frame_id": 20, "score": 0.8}],
        [{"frame_id": 30, "score": 0.7}],
    ]
    chains = align_event_candidates(event_lists, temporal_edges=[(0, 1), (1, 2)])
    assert len(chains) == 1
    assert [event["frame_id"] for event in chains[0]["events"]] == [10, 20, 30]
    assert align_event_candidates(event_lists[:2] + [[]]) == []
    assert [event["frame_id"] for event in align_event_candidates(event_lists)[0]["events"]] == [10, 5, 30]


def test_frame_id_must_be_integer():
    assert normalize_frame_id("001") == 1
    with pytest.raises(ValueError):
        normalize_frame_id("1.5")


def test_csv_writer_quotes_unicode_commas_quotes_and_newlines(tmp_path: Path):
    path = tmp_path / "answer.csv"
    write_csv(path, [["L21_V001", 10, 'Có, "năm" người\ntrên sân khấu']])
    assert read_csv_file(path) == [["L21_V001", "10", 'Có, "năm" người\ntrên sân khấu']]


def test_rrf_fusion_accumulates_each_ranked_list():
    fuser = ScoreFuser(k=60)
    result = fuser.fuse_rrf([
        ("clip", [{"faiss_idx": 1, "score": 0.9}, {"faiss_idx": 2, "score": 0.8}]),
        ("paraphrase", [{"faiss_idx": 2, "score": 0.9}, {"faiss_idx": 1, "score": 0.8}]),
    ])
    assert result[0]["faiss_idx"] in {1, 2}
    assert len(result) == 2
    assert result[0]["score"] == pytest.approx(result[1]["score"])
