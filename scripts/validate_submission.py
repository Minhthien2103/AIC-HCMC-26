#!/usr/bin/env python3
"""Validate AIC2026 CSVs or a Codabench result ZIP."""

from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from src.submission.io import parse_csv_text, read_csv_file, validate_rows  # noqa: E402
from src.submission.query_parser import QuerySpec, load_query_specs  # noqa: E402


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--zip", dest="zip_path", type=Path)
    source.add_argument("--submission-dir", type=Path)
    parser.add_argument("--queries-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--max-rows", type=int, default=100)
    return parser.parse_args()


def _validate_files(specs: list[QuerySpec], readers: dict[str, list[list[str]]], max_rows: int) -> list[str]:
    errors: list[str] = []
    expected = {f"{spec.query_id}.csv": spec for spec in specs}
    missing = sorted(set(expected) - set(readers))
    extra = sorted(set(readers) - set(expected))
    errors.extend(f"missing CSV: {name}" for name in missing)
    errors.extend(f"unexpected CSV: {name}" for name in extra)
    for name, spec in expected.items():
        if name in readers:
            errors.extend(f"{name}: {error}" for error in validate_rows(readers[name], spec, max_rows))
    return errors


def main() -> int:
    args = _args()
    if not 1 <= args.max_rows <= 100:
        raise SystemExit("--max-rows must be between 1 and 100")
    if not args.queries_dir and not args.manifest:
        raise SystemExit("--queries-dir or --manifest is required to determine schemas")
    specs = load_query_specs(args.queries_dir, args.manifest)
    readers: dict[str, list[list[str]]] = {}

    if args.submission_dir:
        if not args.submission_dir.is_dir():
            raise SystemExit(f"Submission directory not found: {args.submission_dir}")
        for path in args.submission_dir.glob("*.csv"):
            try:
                readers[path.name] = read_csv_file(path)
            except UnicodeDecodeError as exc:
                readers[path.name] = [[f"encoding error: {exc}"]]
    else:
        with zipfile.ZipFile(args.zip_path, "r") as archive:
            if archive.testzip() is not None:
                raise SystemExit("ZIP integrity check failed")
            names = [name for name in archive.namelist() if not name.endswith("/")]
            bad_names = [name for name in names if not name.startswith("submission/")]
            if bad_names:
                raise SystemExit("ZIP contains files outside submission/: " + ", ".join(bad_names))
            for name in names:
                relative = name.removeprefix("submission/")
                if "/" in relative or not relative.endswith(".csv"):
                    raise SystemExit(f"Invalid archive member: {name}")
                try:
                    with archive.open(name, "r") as raw:
                        readers[relative] = parse_csv_text(io.TextIOWrapper(raw, encoding="utf-8", newline=""))
                except UnicodeDecodeError as exc:
                    readers[relative] = [[f"encoding error: {exc}"]]

    errors = _validate_files(specs, readers, args.max_rows)
    if errors:
        print("VALIDATION FAILED")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    print(f"VALIDATION PASSED: {len(specs)} query CSV(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
