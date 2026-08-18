#!/usr/bin/env python3
"""Generate and package AIC2026 result CSVs from a query pack."""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from src import config  # noqa: E402
from src.online_pipeline.object_filter import ObjectFilter  # noqa: E402
from src.online_pipeline.query_encoder import QueryEncoder  # noqa: E402
from src.online_pipeline.retrieval import RetrievalEngine  # noqa: E402
from src.online_pipeline.vlm_pipeline import VLMPipeline  # noqa: E402
from src.submission.formatting import format_kis_row, format_qa_row, format_trake_row  # noqa: E402
from src.submission.io import validate_rows, write_csv  # noqa: E402
from src.submission.packaging import package_submission  # noqa: E402
from src.submission.query_parser import QuerySpec, load_query_specs  # noqa: E402
from src.tasks.kis_t import KIStask  # noqa: E402
from src.tasks.trake import TrakeTask  # noqa: E402
from src.tasks.vqa import VQATask  # noqa: E402


LOGGER = logging.getLogger("aic2026.submission")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--queries-dir", type=Path)
    source.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="Output result ZIP")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--max-rows", type=int, default=100)
    parser.add_argument("--vqa-top-k", type=int, default=100)
    parser.add_argument("--trake-top-k", type=int, default=100)
    parser.add_argument("--trake-top-videos", type=int, default=10)
    parser.add_argument("--allow-external-search", action="store_true")
    return parser.parse_args()


def _dedupe_rows(rows: list[list[object]]) -> list[list[object]]:
    seen: set[tuple[object, ...]] = set()
    unique: list[list[object]] = []
    for row in rows:
        key = tuple(row)
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


def _build_tasks(args: argparse.Namespace):
    repo_root = args.repo_root.resolve()
    if not (repo_root / "src" / "config.py").exists():
        raise FileNotFoundError(f"Invalid --repo-root: {repo_root}")
    config.BASE_DIR = repo_root
    config.DATA_DIR = repo_root / "data"
    config.INDEX_DIR = repo_root / "indexes"
    config.VIDEOS_DIR = config.DATA_DIR / "raw_videos"
    config.KEYFRAMES_DIR = config.DATA_DIR / "keyframes"
    config.CLIP_FEATURES_DIR = config.DATA_DIR / "clip-features-32"
    config.MAP_KEYFRAMES_DIR = config.DATA_DIR / "map-keyframes"
    config.OBJECTS_DIR = config.DATA_DIR / "objects"
    config.MEDIA_INFO_DIR = config.DATA_DIR / "media-info"
    config.FAISS_INDEX_PATH = config.INDEX_DIR / "faiss_clip.index"
    config.METADATA_PATH = config.INDEX_DIR / "metadata.parquet"
    config.OBJECTS_PATH = config.INDEX_DIR / "objects.parquet"
    config.MEDIA_INFO_PATH = config.INDEX_DIR / "media_info.json"
    encoder = QueryEncoder(config.CLIP_MODEL_NAME, config.CLIP_PRETRAINED, device=args.device)
    retriever = RetrievalEngine(config.FAISS_INDEX_PATH, config.METADATA_PATH)
    if retriever.index is None or retriever.meta_df is None:
        raise RuntimeError("FAISS index and metadata are required before generating a submission")
    object_filter = ObjectFilter(config.OBJECTS_PATH)
    vlm = VLMPipeline(device=args.device)
    kis = KIStask(encoder, retriever, object_filter=object_filter)
    trake = TrakeTask(encoder, retriever, vlm_pipeline=vlm)
    # SemanticObjectFilter is optional and lazy; the VQA task falls back to
    # exact labels when spaCy/SentenceTransformers are not installed.
    semantic_filter = None
    try:
        from src.online_pipeline.semantic_object_filter import SemanticObjectFilter

        semantic_filter = SemanticObjectFilter(object_filter.get_all_labels())
    except Exception as exc:
        LOGGER.warning("Semantic object filter disabled: %s", exc)
    vqa = VQATask(encoder, retriever, object_filter=object_filter, vlm_pipeline=vlm, semantic_filter=semantic_filter)
    return kis, vqa, trake


def _generate_for_query(spec: QuerySpec, tasks, args: argparse.Namespace) -> list[list[object]]:
    kis, vqa, trake = tasks
    if spec.query_type == "kis":
        results = kis.execute(spec.description, top_k=args.max_rows)
        rows = []
        for result in results:
            try:
                rows.append(format_kis_row(result))
            except (KeyError, ValueError) as exc:
                LOGGER.warning("Skipping KIS candidate for %s: %s", spec.query_id, exc)
        return _dedupe_rows(rows)[: args.max_rows]
    if spec.query_type == "qa":
        results, _analysis = vqa.execute(
            spec.question or spec.description,
            top_k=min(args.vqa_top_k, args.max_rows),
            allow_external_search=args.allow_external_search,
        )
        rows = []
        for result in results:
            try:
                rows.append(format_qa_row(result, max_chars=config.VQA_MAX_ANSWER_CHARS))
            except (KeyError, ValueError) as exc:
                LOGGER.warning("Skipping VQA candidate for %s: %s", spec.query_id, exc)
        return _dedupe_rows(rows)[: args.max_rows]

    results = trake.execute(
        spec.description,
        list(spec.events),
        top_videos=args.trake_top_videos,
        max_sequences=min(args.trake_top_k, args.max_rows),
    )
    rows = []
    for result in results:
        try:
            rows.append(format_trake_row(result))
        except (KeyError, ValueError) as exc:
            LOGGER.warning("Skipping TRAKE candidate for %s: %s", spec.query_id, exc)
    return _dedupe_rows(rows)[: args.max_rows]


def main() -> int:
    args = _parse_args()
    if not 1 <= args.max_rows <= 100:
        raise SystemExit("--max-rows must be between 1 and 100")
    if not 1 <= args.vqa_top_k <= 100 or not 1 <= args.trake_top_k <= 100:
        raise SystemExit("--vqa-top-k and --trake-top-k must be between 1 and 100")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    specs = load_query_specs(args.queries_dir, args.manifest)
    tasks = _build_tasks(args)
    with tempfile.TemporaryDirectory(prefix="aic2026-submission-") as temp_dir:
        submission_dir = Path(temp_dir) / "submission"
        for spec in specs:
            rows = _generate_for_query(spec, tasks, args)
            if not rows:
                raise RuntimeError(f"Query {spec.query_id} produced no valid submission rows")
            errors = validate_rows([[str(value) for value in row] for row in rows], spec, args.max_rows)
            if errors:
                raise RuntimeError(f"Generated CSV for {spec.query_id} failed validation: {'; '.join(errors)}")
            output_csv = submission_dir / f"{spec.query_id}.csv"
            write_csv(output_csv, rows)
            LOGGER.info("%s: wrote %d rows", spec.query_id, len(rows))
        package_submission(submission_dir, args.output)
    LOGGER.info("Created %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
