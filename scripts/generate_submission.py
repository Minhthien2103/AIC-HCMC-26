#!/usr/bin/env python3
"""Generate and package AIC2026 result CSVs from a query pack."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from src import config  # noqa: E402
from src.online_pipeline.object_filter import ObjectFilter  # noqa: E402
from src.online_pipeline.query_encoder import QueryEncoder  # noqa: E402
from src.online_pipeline.retrieval import RetrievalEngine  # noqa: E402
from src.online_pipeline.text_evidence import (  # noqa: E402
    CandidateOCR,
    E5TextEncoder,
    MediaTextRetriever,
    OfflineEvidenceCache,
)
from src.online_pipeline.vlm_pipeline import VLMPipeline  # noqa: E402
from src.submission.formatting import format_kis_row, format_qa_row, format_trake_row  # noqa: E402
from src.submission.io import validate_rows, write_csv  # noqa: E402
from src.submission.packaging import package_submission  # noqa: E402
from src.submission.query_parser import QuerySpec, load_query_specs  # noqa: E402
from src.submission.review import apply_review, write_review_assets, write_review_manifest_template  # noqa: E402
from src.tasks.kis_t import KIStask  # noqa: E402
from src.tasks.trake import TrakeTask  # noqa: E402
from src.tasks.vqa import VQATask  # noqa: E402
from src.utils.translation import configure_translation  # noqa: E402


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
    parser.add_argument(
        "--disable-kis-qwen",
        action="store_true",
        help="Disable Qwen2-VL KIS query analysis and visual re-ranking.",
    )
    parser.add_argument(
        "--kis-vlm-top-k",
        type=int,
        default=config.KIS_QWEN_RERANK_TOP_K,
        help="Maximum KIS candidates visually scored by Qwen2-VL (1-100).",
    )
    parser.add_argument(
        "--enable-kis-dual",
        action="store_true",
        help="Use ViT-H/14, BTC media E5 and candidate OCR alongside the ViT-B baseline.",
    )
    parser.add_argument(
        "--require-kis-assets",
        action="store_true",
        help="Fail instead of silently falling back when requested KIS assets/caches are missing.",
    )
    parser.add_argument("--offline", action="store_true", help="Forbid network-dependent KIS translation/evidence.")
    parser.add_argument("--translation-cache", type=Path)
    parser.add_argument("--evidence-cache", type=Path)
    parser.add_argument("--ocr-cache-dir", type=Path)
    parser.add_argument("--disable-kis-ocr", action="store_true")
    parser.add_argument("--kis-candidate-budget", type=int, default=config.KIS_DUAL_CANDIDATE_BUDGET)
    parser.add_argument("--kis-ocr-candidate-budget", type=int, default=config.KIS_OCR_CANDIDATE_BUDGET)
    parser.add_argument("--kis-text-frames-per-video", type=int, default=config.KIS_TEXT_FRAMES_PER_VIDEO)
    parser.add_argument("--review-output-dir", type=Path, help="Write KIS top-20 contact sheets and provenance here.")
    parser.add_argument("--review-manifest", type=Path, help="JSON pin/keep/reject decisions for generated candidates.")
    parser.add_argument("--review-top-k", type=int, default=config.KIS_REVIEW_TOP_K)
    parser.add_argument("--provenance-dir", type=Path, help="Directory for reproducibility metadata (default next to ZIP).")
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


def _build_tasks(args: argparse.Namespace, specs: list[QuerySpec]):
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
    config.VITH_INDEX_PATH = config.INDEX_DIR / "faiss_vith.index"
    config.VITH_FEATURES_PATH = config.INDEX_DIR / "vith_features.f16.npy"
    config.MEDIA_TEXT_INDEX_PATH = config.INDEX_DIR / "media_e5.index"
    config.MEDIA_TEXT_RECORDS_PATH = config.INDEX_DIR / "media_e5_records.json"
    if args.offline:
        # Hugging Face consumers honour these before attempting a download;
        # missing model files therefore fail visibly instead of contacting the
        # network during an audited final run.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["PADDLE_PDX_MODEL_SOURCE"] = "local"
    translation_cache = args.translation_cache or (config.INDEX_DIR / "translation_mbart_cache.json")
    # mBART handles only a few query strings; keep it on CPU so the T4 has
    # headroom for ViT-H, E5 and Qwen2-VL during candidate reranking.
    configure_translation(cache_path=translation_cache, offline=args.offline, device="cpu")
    encoder = QueryEncoder(config.CLIP_MODEL_NAME, config.CLIP_PRETRAINED, device=args.device)
    retriever = RetrievalEngine(config.FAISS_INDEX_PATH, config.METADATA_PATH)
    if retriever.index is None or retriever.meta_df is None:
        raise RuntimeError("FAISS index and metadata are required before generating a submission")
    object_filter = ObjectFilter(config.OBJECTS_PATH)

    needs_vqa = any(spec.query_type == "qa" for spec in specs)
    needs_trake = any(spec.query_type == "trake" for spec in specs)
    kis_qwen_enabled = not args.disable_kis_qwen and args.device == "cuda"
    if not args.disable_kis_qwen and args.device != "cuda":
        LOGGER.warning("KIS Qwen is disabled because --device=%s is not CUDA", args.device)

    # This object is lazy: Qwen weights load only if a VQA/TRAKE/KIS visual
    # operation actually needs them. A KIS-only CPU run therefore remains CLIP
    # only, while a CUDA KIS run reuses the same loaded 4-bit Qwen instance.
    vlm = VLMPipeline(device=args.device, local_files_only=args.offline) if (kis_qwen_enabled or needs_vqa or needs_trake) else None
    has_kis = any(spec.query_type == "kis" for spec in specs)
    evidence_path = args.evidence_cache or (config.INDEX_DIR / "kis_evidence_cache.json")
    evidence_cache = None
    if has_kis and (args.offline or args.evidence_cache is not None):
        evidence_cache = OfflineEvidenceCache(evidence_path, required=args.offline)
        if args.offline:
            # Fail before expensive model loading if the prepared cache does
            # not cover every requested KIS query.
            for spec in specs:
                if spec.query_type == "kis":
                    evidence_cache.context(spec.query_id)

    secondary_retrievers = {}
    media_retriever = None
    ocr = None
    if args.enable_kis_dual and has_kis:
        missing_assets = [
            path for path in (config.VITH_INDEX_PATH, config.MEDIA_TEXT_INDEX_PATH, config.MEDIA_TEXT_RECORDS_PATH)
            if not path.exists()
        ]
        if missing_assets:
            message = "KIS dual assets are missing: " + ", ".join(str(path) for path in missing_assets)
            if args.require_kis_assets:
                raise FileNotFoundError(message)
            LOGGER.warning("%s; using the ViT-B baseline only", message)
        else:
            vith_encoder = QueryEncoder(config.VITH_MODEL_NAME, config.VITH_PRETRAINED, device=args.device)
            vith_retriever = RetrievalEngine(config.VITH_INDEX_PATH, config.METADATA_PATH)
            if vith_retriever.index is None or vith_retriever.meta_df is None:
                raise RuntimeError("ViT-H index exists but could not be loaded")
            e5_encoder = E5TextEncoder(config.E5_MODEL_NAME, device=args.device, local_files_only=True)
            if args.require_kis_assets:
                # Force an immediate, local-only E5 load so final mode does
                # not discover a missing model after expensive CLIP/Qwen work.
                e5_encoder.encode_queries(["asset availability check"])
            secondary_retrievers["clip_vith14"] = (vith_encoder, vith_retriever)
            media_retriever = MediaTextRetriever(config.MEDIA_TEXT_INDEX_PATH, config.MEDIA_TEXT_RECORDS_PATH, e5_encoder)
            if not args.disable_kis_ocr:
                ocr = CandidateOCR(args.ocr_cache_dir or (config.INDEX_DIR / "ocr_cache"), e5_encoder)

    kis = KIStask(
        encoder,
        retriever,
        object_filter=object_filter,
        vlm_pipeline=vlm,
        enable_qwen=kis_qwen_enabled,
        vlm_top_k=args.kis_vlm_top_k,
        secondary_retrievers=secondary_retrievers,
        media_retriever=media_retriever,
        ocr=ocr,
        evidence_cache=evidence_cache,
        candidate_budget=args.kis_candidate_budget,
        ocr_candidate_budget=args.kis_ocr_candidate_budget,
        text_frames_per_video=args.kis_text_frames_per_video,
        strict_sources=args.require_kis_assets,
    )
    trake = TrakeTask(encoder, retriever, vlm_pipeline=vlm) if needs_trake else None

    vqa = None
    if needs_vqa:
        # SemanticObjectFilter is optional and lazy; the VQA task falls back to
        # exact labels when spaCy/SentenceTransformers are not installed.
        semantic_filter = None
        try:
            from src.online_pipeline.semantic_object_filter import SemanticObjectFilter

            semantic_filter = SemanticObjectFilter(object_filter.get_all_labels())
        except Exception as exc:
            LOGGER.warning("Semantic object filter disabled: %s", exc)
        vqa = VQATask(
            encoder,
            retriever,
            object_filter=object_filter,
            vlm_pipeline=vlm,
            semantic_filter=semantic_filter,
        )
    return kis, vqa, trake


def _generate_for_query(spec: QuerySpec, tasks, args: argparse.Namespace) -> list[list[object]]:
    kis, vqa, trake = tasks
    if spec.query_type == "kis":
        results = kis.execute(spec.description, top_k=args.max_rows, query_id=spec.query_id)
        if args.review_output_dir is not None:
            review_dir = write_review_assets(
                spec.query_id,
                results,
                args.review_output_dir,
                image_path_for=lambda item: config.keyframe_path(item["video_id"], item["keyframe_name"]),
                limit=args.review_top_k,
            )
            LOGGER.info("%s: wrote review artefacts to %s", spec.query_id, review_dir)
        results = apply_review(
            results,
            spec.query_id,
            args.review_manifest,
            limit=args.review_top_k,
        )
        rows = []
        for result in results:
            try:
                rows.append(format_kis_row(result))
            except (KeyError, ValueError) as exc:
                LOGGER.warning("Skipping KIS candidate for %s: %s", spec.query_id, exc)
        return _dedupe_rows(rows)[: args.max_rows]
    if spec.query_type == "qa":
        if vqa is None:
            raise RuntimeError("VQA task was not initialized")
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

    if trake is None:
        raise RuntimeError("TRAKE task was not initialized")
    if not spec.events:
        raise RuntimeError(
            f"TRAKE query {spec.query_id} has no parsed event structure; refusing to let Qwen guess event count. "
            "Fix the query parser or use a manifest with explicit events."
        )
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


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_provenance(args: argparse.Namespace, specs: list[QuerySpec]) -> Path:
    directory = args.provenance_dir or args.output.with_suffix("").with_name(f"{args.output.stem}_provenance")
    directory.mkdir(parents=True, exist_ok=True)
    relevant_paths = {
        "vitb_index": config.FAISS_INDEX_PATH,
        "metadata": config.METADATA_PATH,
        "vith_index": config.VITH_INDEX_PATH,
        "media_e5_index": config.MEDIA_TEXT_INDEX_PATH,
        "media_e5_records": config.MEDIA_TEXT_RECORDS_PATH,
        "translation_cache": args.translation_cache or (config.INDEX_DIR / "translation_mbart_cache.json"),
        "evidence_cache": args.evidence_cache or (config.INDEX_DIR / "kis_evidence_cache.json"),
        "review_manifest": args.review_manifest,
    }
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_zip": str(args.output.resolve()),
        "query_ids": [spec.query_id for spec in specs],
        "models": {
            "vitb": {"name": config.CLIP_MODEL_NAME, "pretrained": config.CLIP_PRETRAINED},
            "vith": {"name": config.VITH_MODEL_NAME, "pretrained": config.VITH_PRETRAINED},
            "e5": config.E5_MODEL_NAME,
            "qwen": "Qwen/Qwen2-VL-7B-Instruct",
            "translation": "facebook/mbart-large-50-many-to-many-mmt",
        },
        "config": {
            "offline": args.offline,
            "enable_kis_dual": args.enable_kis_dual,
            "candidate_budget": args.kis_candidate_budget,
            "ocr_candidate_budget": args.kis_ocr_candidate_budget,
            "qwen_top_k": args.kis_vlm_top_k,
            "rrf_k": config.KIS_RRF_K,
            "review_top_k": args.review_top_k,
        },
        "asset_sha256": {
            name: _sha256(Path(path)) if path is not None else None
            for name, path in relevant_paths.items()
        },
        "candidate_evidence": {
            "review_output_dir": str(args.review_output_dir) if args.review_output_dir is not None else None,
            "ocr_cache_dir": str(args.ocr_cache_dir or (config.INDEX_DIR / "ocr_cache")),
            "ocr_cache_entries": len(list((args.ocr_cache_dir or (config.INDEX_DIR / "ocr_cache")).glob("*.json"))),
        },
    }
    provenance_path = directory / "run_provenance.json"
    provenance_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return provenance_path


def main() -> int:
    args = _parse_args()
    if not 1 <= args.max_rows <= 100:
        raise SystemExit("--max-rows must be between 1 and 100")
    if not 1 <= args.vqa_top_k <= 100 or not 1 <= args.trake_top_k <= 100:
        raise SystemExit("--vqa-top-k and --trake-top-k must be between 1 and 100")
    if not 1 <= args.kis_vlm_top_k <= 100:
        raise SystemExit("--kis-vlm-top-k must be between 1 and 100")
    if min(args.kis_candidate_budget, args.kis_ocr_candidate_budget, args.kis_text_frames_per_video, args.review_top_k) < 1:
        raise SystemExit("KIS candidate/review budgets must be positive")
    if args.offline and args.allow_external_search:
        raise SystemExit("--offline and --allow-external-search cannot be used together")
    if args.provenance_dir is None:
        args.provenance_dir = args.output.with_suffix("").with_name(f"{args.output.stem}_provenance")
    if args.review_output_dir is None:
        # Candidate-level evidence is part of every reproducible run, even
        # when the caller is not yet ready to make manual decisions.
        args.review_output_dir = args.provenance_dir / "kis_review"
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    specs = load_query_specs(args.queries_dir, args.manifest)
    tasks = _build_tasks(args, specs)
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
    if args.review_output_dir is not None:
        template = write_review_manifest_template(
            args.review_output_dir,
            [spec.query_id for spec in specs if spec.query_type == "kis"],
        )
        LOGGER.info("Wrote editable review template %s", template)
    LOGGER.info("Wrote reproducibility record %s", _write_provenance(args, specs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
