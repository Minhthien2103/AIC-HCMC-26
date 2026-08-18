#!/usr/bin/env python3
"""Validate and package a directory of AIC result CSVs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from src.submission.io import read_csv_file, validate_rows  # noqa: E402
from src.submission.packaging import package_submission  # noqa: E402
from src.submission.query_parser import load_query_specs  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queries-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--max-rows", type=int, default=100)
    args = parser.parse_args()
    if not 1 <= args.max_rows <= 100:
        raise SystemExit("--max-rows must be between 1 and 100")
    if not args.queries_dir and not args.manifest:
        raise SystemExit("--queries-dir or --manifest is required")
    specs = load_query_specs(args.queries_dir, args.manifest)
    errors = []
    expected_names = {f"{spec.query_id}.csv" for spec in specs}
    actual_names = {path.name for path in args.submission_dir.glob("*.csv")}
    errors.extend(f"unexpected CSV: {name}" for name in sorted(actual_names - expected_names))
    for spec in specs:
        path = args.submission_dir / f"{spec.query_id}.csv"
        if not path.exists():
            errors.append(f"missing CSV: {path.name}")
            continue
        try:
            errors.extend(f"{path.name}: {error}" for error in validate_rows(read_csv_file(path), spec, args.max_rows))
        except UnicodeDecodeError:
            errors.append(f"{path.name}: file is not valid UTF-8")
    if errors:
        raise SystemExit("Cannot package invalid submission:\n" + "\n".join(errors))
    output = package_submission(args.submission_dir, args.output)
    print(f"Created {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
