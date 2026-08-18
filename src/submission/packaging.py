"""Create Codabench result archives with the required submission/ root."""

from __future__ import annotations

import zipfile
from pathlib import Path


def package_submission(submission_dir: str | Path, output_zip: str | Path) -> Path:
    source = Path(submission_dir)
    output = Path(output_zip)
    if not source.is_dir():
        raise FileNotFoundError(f"Submission directory not found: {source}")
    csv_files = sorted(source.glob("*.csv"))
    if not csv_files:
        raise ValueError("No CSV files found in submission directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in csv_files:
            archive.write(path, arcname=f"submission/{path.name}")
    with zipfile.ZipFile(output, "r") as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP integrity check failed")
        names = archive.namelist()
        if not names or any(not name.startswith("submission/") for name in names):
            raise ValueError("Every ZIP member must be inside submission/")
    return output
