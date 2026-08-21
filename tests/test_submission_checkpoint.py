from __future__ import annotations

import sys
import zipfile
from pathlib import Path

from scripts import generate_submission
from src.submission.io import write_csv


def test_generate_submission_resumes_valid_checkpoint_and_packages_exact_pack(
    tmp_path: Path,
    monkeypatch,
):
    queries = tmp_path / "queries"
    queries.mkdir()
    (queries / "query-p1-1-kis.txt").write_text("Một người đang chèo thuyền.", encoding="utf-8")

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    write_csv(checkpoint / "query-p1-1-kis.csv", [["L21_V001", 123]])
    write_csv(checkpoint / "stale-query.csv", [["L99_V999", 999]])
    output = tmp_path / "result.zip"

    monkeypatch.setattr(generate_submission, "_build_tasks", lambda args, specs: object())

    def unexpected_generation(*_args, **_kwargs):
        raise AssertionError("a valid checkpoint must not be regenerated")

    monkeypatch.setattr(generate_submission, "_generate_for_query", unexpected_generation)
    monkeypatch.setattr(
        generate_submission,
        "write_review_manifest_template",
        lambda *_args, **_kwargs: tmp_path / "review-template.json",
    )
    monkeypatch.setattr(
        generate_submission,
        "_write_provenance",
        lambda *_args, **_kwargs: tmp_path / "provenance.json",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_submission.py",
            "--queries-dir",
            str(queries),
            "--output",
            str(output),
            "--checkpoint-dir",
            str(checkpoint),
            "--resume",
            "--device",
            "cpu",
            "--disable-kis-qwen",
        ],
    )

    assert generate_submission.main() == 0
    with zipfile.ZipFile(output) as archive:
        assert archive.namelist() == ["submission/query-p1-1-kis.csv"]
