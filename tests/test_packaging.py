from pathlib import Path
import zipfile

from src.submission.packaging import package_submission


def test_package_has_submission_root(tmp_path: Path):
    source = tmp_path / "submission"
    source.mkdir()
    (source / "query-1-kis.csv").write_text("L21_V001,10\n", encoding="utf-8")
    output = package_submission(source, tmp_path / "submission.zip")
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        assert archive.namelist() == ["submission/query-1-kis.csv"]
