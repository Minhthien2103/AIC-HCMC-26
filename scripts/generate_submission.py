#!/usr/bin/env python3
"""Generate and package AIC2026 result CSVs from a query pack."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from src import config  # noqa: E402
from src.online_pipeline.object_filter import ObjectFilter  # noqa: E402
from src.online_pipeline.frame_neighborhood import FrameNeighborhood  # noqa: E402
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
from src.submission.io import read_csv_file, validate_rows, write_csv  # noqa: E402
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
    parser.add_argument(
        "--retrieval-backend",
        choices=("zilliz", "faiss"),
        default=config.RETRIEVAL_BACKEND,
        help="zilliz searches MobileCLIP visual+subtitle collections; faiss keeps the legacy local index.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        help="Canonical mapping metadata.parquet (or set AIC_METADATA_PATH).",
    )
    parser.add_argument(
        "--keyframes-dir",
        type=Path,
        help="Extracted keyframe root used by Qwen/OCR (or set AIC_KEYFRAMES_DIR).",
    )
    parser.add_argument("--zilliz-visual-collection", default=config.ZILLIZ_VISUAL_COLLECTION)
    parser.add_argument("--zilliz-text-collection", default=config.ZILLIZ_TEXT_COLLECTION)
    parser.add_argument("--zilliz-object-collection", default=config.ZILLIZ_OBJECT_COLLECTION)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument(
        "--vlm-mode",
        choices=("4bit", "bf16"),
        default="4bit",
        help="Qwen load mode. Use bf16 on A100 for higher throughput; 4bit uses less VRAM.",
    )
    parser.add_argument("--max-rows", type=int, default=100)
    parser.add_argument("--vqa-top-k", type=int, default=100)
    parser.add_argument(
        "--vqa-qwen-candidate-budget",
        type=int,
        default=config.VQA_QWEN_CANDIDATE_BUDGET,
        help="Maximum video-stratified QA frames inspected by Qwen2-VL.",
    )
    parser.add_argument("--trake-top-k", type=int, default=100)
    parser.add_argument("--trake-top-videos", type=int, default=10)
    parser.add_argument("--trake-event-top-k", type=int, default=config.TRAKE_EVENT_TOP_K)
    parser.add_argument("--trake-qwen-per-event", type=int, default=config.TRAKE_QWEN_PER_EVENT)
    parser.add_argument(
        "--disable-kis-qwen",
        action="store_true",
        help="Disable Qwen2-VL KIS query analysis and visual re-ranking.",
    )
    parser.add_argument(
        "--kis-vlm-top-k",
        type=int,
        default=config.KIS_QWEN_RERANK_TOP_K,
        help="Maximum stratified KIS candidates visually scored by Qwen2-VL.",
    )
    parser.add_argument(
        "--kis-profile",
        choices=("fast", "full"),
        default="fast",
        help="fast=primary retrieval+optional media-E5/OCR/Qwen; full additionally requires the legacy ViT-H index.",
    )
    parser.add_argument(
        "--kis-baseline-simple",
        action="store_true",
        help="Use HCMAI-style raw-CLIP video aggregation and top-k-aware output allocation.",
    )
    parser.add_argument(
        "--enable-kis-dual",
        action="store_true",
        help="Deprecated alias for --kis-profile full.",
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
    parser.add_argument("--kis-video-budget", type=int, default=config.KIS_VIDEO_BUDGET)
    parser.add_argument("--kis-local-frame-budget", type=int, default=config.KIS_LOCAL_FRAME_BUDGET)
    parser.add_argument("--kis-query-variant-limit", type=int, default=config.KIS_QUERY_VARIANT_LIMIT)
    parser.add_argument("--kis-text-frames-per-video", type=int, help="Deprecated alias for --kis-local-frame-budget.")
    parser.add_argument("--frame-neighborhood-count", type=int, default=config.FRAME_NEIGHBORHOOD_COUNT)
    parser.add_argument("--review-output-dir", type=Path, help="Write top-20 review contact sheets and provenance for all query types here.")
    parser.add_argument("--review-manifest", type=Path, help="JSON pin/keep/reject decisions for generated candidates.")
    parser.add_argument("--review-top-k", type=int, default=config.KIS_REVIEW_TOP_K)
    parser.add_argument("--provenance-dir", type=Path, help="Directory for reproducibility metadata (default next to ZIP).")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Persist validated per-query CSVs here so an interrupted batch can resume.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse valid CSVs already present in --checkpoint-dir.",
    )
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
    config.KEYFRAMES_DIR = (
        args.keyframes_dir.expanduser().resolve()
        if args.keyframes_dir is not None
        else Path(os.getenv("AIC_KEYFRAMES_DIR", config.DATA_DIR / "keyframes")).expanduser()
    )
    config.CLIP_FEATURES_DIR = config.DATA_DIR / "clip-features-32"
    config.MAP_KEYFRAMES_DIR = config.DATA_DIR / "map-keyframes"
    config.OBJECTS_DIR = config.DATA_DIR / "objects"
    config.MEDIA_INFO_DIR = config.DATA_DIR / "media-info"
    config.FAISS_INDEX_PATH = config.INDEX_DIR / "faiss_clip.index"
    config.METADATA_PATH = (
        args.metadata_path.expanduser().resolve()
        if args.metadata_path is not None
        else Path(os.getenv("AIC_METADATA_PATH", config.INDEX_DIR / "metadata.parquet")).expanduser()
    )
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
    translation_cache = args.translation_cache or (config.INDEX_DIR / "translation_mbart_cache.json")
    # mBART handles only a few query strings; keep it on CPU so the T4 has
    # headroom for ViT-H, E5 and Qwen2-VL during candidate reranking.
    configure_translation(cache_path=translation_cache, offline=args.offline, device="cpu")
    if args.retrieval_backend == "zilliz":
        from src.online_pipeline.zilliz_object_filter import ZillizObjectFilter
        from src.online_pipeline.zilliz_retrieval import ZillizRetrievalEngine

        encoder = QueryEncoder(
            config.ZILLIZ_CLIP_MODEL_NAME,
            config.ZILLIZ_CLIP_PRETRAINED,
            device=args.device,
        )
        retriever = ZillizRetrievalEngine(
            uri=os.getenv("ZILLIZ_URI", config.ZILLIZ_URI),
            token=os.getenv("ZILLIZ_TOKEN", config.ZILLIZ_TOKEN),
            metadata_path=config.METADATA_PATH,
            visual_collection=args.zilliz_visual_collection,
            text_collection=args.zilliz_text_collection,
            vector_field=config.ZILLIZ_VECTOR_FIELD,
            visual_pk_field=config.ZILLIZ_VISUAL_PK_FIELD,
            text_pk_field=config.ZILLIZ_TEXT_PK_FIELD,
            metric_type=config.ZILLIZ_METRIC_TYPE,
            dimension=config.ZILLIZ_CLIP_DIM,
        )
        object_filter = ZillizObjectFilter(
            retriever.client,
            args.zilliz_object_collection,
            retriever.meta_df,
            pk_field=config.ZILLIZ_OBJECT_PK_FIELD,
        )
        LOGGER.info(
            "Using Zilliz collections visual=%s text=%s object=%s with metadata=%s",
            args.zilliz_visual_collection,
            args.zilliz_text_collection,
            args.zilliz_object_collection,
            config.METADATA_PATH,
        )
    else:
        encoder = QueryEncoder(config.CLIP_MODEL_NAME, config.CLIP_PRETRAINED, device=args.device)
        retriever = RetrievalEngine(config.FAISS_INDEX_PATH, config.METADATA_PATH)
        if retriever.index is None or retriever.meta_df is None:
            raise RuntimeError("FAISS index and metadata are required before generating a submission")
        object_filter = ObjectFilter(config.OBJECTS_PATH)
    frame_neighborhood = FrameNeighborhood(retriever.meta_df)

    needs_vqa = any(spec.query_type == "qa" for spec in specs)
    needs_trake = any(spec.query_type == "trake" for spec in specs)
    kis_qwen_enabled = not args.disable_kis_qwen and args.device == "cuda"
    if not args.disable_kis_qwen and args.device != "cuda":
        LOGGER.warning("KIS Qwen is disabled because --device=%s is not CUDA", args.device)

    # This object is lazy: Qwen weights load only if a VQA/TRAKE/KIS visual
    # operation actually needs them. A KIS-only CPU run therefore remains CLIP
    # only, while a CUDA KIS run reuses the same loaded Qwen instance.
    vlm = VLMPipeline(
        device=args.device,
        local_files_only=args.offline,
        load_mode=args.vlm_mode,
    ) if (kis_qwen_enabled or needs_vqa or needs_trake) else None
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
    # Fast profile is fully useful without the expensive ViT-H index.  Media
    # E5/OCR are shared with QA/TRAKE when the prepared BTC assets exist.
    e5_missing = [path for path in (config.MEDIA_TEXT_INDEX_PATH, config.MEDIA_TEXT_RECORDS_PATH) if not path.exists()]
    e5_encoder = None
    if not e5_missing:
        e5_encoder = E5TextEncoder(config.E5_MODEL_NAME, device=args.device, local_files_only=args.offline)
        if args.require_kis_assets:
            e5_encoder.encode_queries(["asset availability check"])
        media_retriever = MediaTextRetriever(config.MEDIA_TEXT_INDEX_PATH, config.MEDIA_TEXT_RECORDS_PATH, e5_encoder)
        if not args.disable_kis_ocr:
            ocr = CandidateOCR(args.ocr_cache_dir or (config.INDEX_DIR / "ocr_cache"), e5_encoder)
    elif has_kis:
        message = "Fast KIS media-E5 assets are missing: " + ", ".join(str(path) for path in e5_missing)
        if args.require_kis_assets:
            raise FileNotFoundError(message)
        LOGGER.warning("%s; retaining primary embedding/Qwen retrieval", message)

    if args.kis_profile == "full" and has_kis and args.retrieval_backend == "zilliz":
        LOGGER.warning("--kis-profile full uses a legacy local index and is disabled with the Zilliz backend")
        args.kis_profile = "fast"
    if args.kis_profile == "full" and has_kis:
        if not config.VITH_INDEX_PATH.exists():
            message = f"KIS full profile requires ViT-H index: {config.VITH_INDEX_PATH}"
            if args.require_kis_assets:
                raise FileNotFoundError(message)
            LOGGER.warning("%s; falling back to fast profile", message)
            args.kis_profile = "fast"
        else:
            vith_encoder = QueryEncoder(config.VITH_MODEL_NAME, config.VITH_PRETRAINED, device=args.device)
            vith_retriever = RetrievalEngine(config.VITH_INDEX_PATH, config.METADATA_PATH)
            if vith_retriever.index is None or vith_retriever.meta_df is None:
                raise RuntimeError("ViT-H index exists but could not be loaded")
            secondary_retrievers["clip_vith14"] = (vith_encoder, vith_retriever)

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
        frame_neighborhood=frame_neighborhood,
        video_budget=args.kis_video_budget,
        local_frame_budget=args.kis_local_frame_budget,
        query_variant_limit=args.kis_query_variant_limit,
        neighborhood_count=args.frame_neighborhood_count,
        strict_sources=args.require_kis_assets,
        baseline_simple=args.kis_baseline_simple,
    )
    trake = TrakeTask(encoder, retriever, vlm_pipeline=vlm, media_retriever=media_retriever, ocr=ocr, frame_neighborhood=frame_neighborhood) if needs_trake else None

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
            media_retriever=media_retriever,
            ocr=ocr,
            frame_neighborhood=frame_neighborhood,
            qwen_candidate_budget=args.vqa_qwen_candidate_budget,
            neighborhood_count=args.frame_neighborhood_count,
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
        if args.review_output_dir is not None:
            review_dir = write_review_assets(
                spec.query_id,
                results,
                args.review_output_dir,
                image_path_for=lambda item: config.keyframe_path(item["video_id"], item["keyframe_name"]),
                limit=args.review_top_k,
            )
            LOGGER.info("%s: wrote review artefacts to %s", spec.query_id, review_dir)
        results = apply_review(results, spec.query_id, args.review_manifest, limit=args.review_top_k)
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
        event_top_k=args.trake_event_top_k,
        max_sequences=min(args.trake_top_k, args.max_rows),
        qwen_per_event=args.trake_qwen_per_event,
        neighborhood_count=args.frame_neighborhood_count,
    )
    if args.review_output_dir is not None:
        review_dir = write_review_assets(
            spec.query_id,
            results,
            args.review_output_dir,
            image_path_for=lambda item: config.keyframe_path(item["video_id"], item["keyframe_name"]),
            limit=args.review_top_k,
        )
        LOGGER.info("%s: wrote review artefacts to %s", spec.query_id, review_dir)
    results = apply_review(results, spec.query_id, args.review_manifest, limit=args.review_top_k)
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
        "vitb_index": config.FAISS_INDEX_PATH if args.retrieval_backend == "faiss" else None,
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
            "primary_retrieval": {
                "name": config.ZILLIZ_CLIP_MODEL_NAME if args.retrieval_backend == "zilliz" else config.CLIP_MODEL_NAME,
                "pretrained": config.ZILLIZ_CLIP_PRETRAINED if args.retrieval_backend == "zilliz" else config.CLIP_PRETRAINED,
            },
            "vith": {"name": config.VITH_MODEL_NAME, "pretrained": config.VITH_PRETRAINED},
            "e5": config.E5_MODEL_NAME,
            "qwen": "Qwen/Qwen2-VL-7B-Instruct",
            "translation": "facebook/mbart-large-50-many-to-many-mmt",
        },
        "config": {
            "retrieval_backend": args.retrieval_backend,
            "zilliz_collections": {
                "visual": args.zilliz_visual_collection,
                "text": args.zilliz_text_collection,
                "object": args.zilliz_object_collection,
            } if args.retrieval_backend == "zilliz" else None,
            "metadata_path": str(config.METADATA_PATH),
            "keyframes_dir": str(config.KEYFRAMES_DIR),
            "offline": args.offline,
            "kis_profile": args.kis_profile,
            "kis_baseline_simple": args.kis_baseline_simple,
            "enable_kis_dual_alias": args.enable_kis_dual,
            "candidate_budget": args.kis_candidate_budget,
            "ocr_candidate_budget": args.kis_ocr_candidate_budget,
            "video_budget": args.kis_video_budget,
            "local_frame_budget": args.kis_local_frame_budget,
            "query_variant_limit": args.kis_query_variant_limit,
            "qwen_budget": args.kis_vlm_top_k,
            "vqa_qwen_candidate_budget": args.vqa_qwen_candidate_budget,
            "vlm_mode": args.vlm_mode,
            "frame_neighborhood_count": args.frame_neighborhood_count,
            "trake_event_top_k": args.trake_event_top_k,
            "trake_qwen_per_event": args.trake_qwen_per_event,
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
    if args.enable_kis_dual:
        args.kis_profile = "full"
    if args.kis_text_frames_per_video is not None:
        args.kis_local_frame_budget = args.kis_text_frames_per_video
    if not 1 <= args.max_rows <= 100:
        raise SystemExit("--max-rows must be between 1 and 100")
    if not 1 <= args.vqa_top_k <= 100 or not 1 <= args.trake_top_k <= 100:
        raise SystemExit("--vqa-top-k and --trake-top-k must be between 1 and 100")
    if args.kis_vlm_top_k < 1:
        raise SystemExit("--kis-vlm-top-k must be positive")
    if min(
        args.kis_candidate_budget,
        args.kis_ocr_candidate_budget,
        args.kis_video_budget,
        args.kis_local_frame_budget,
        args.kis_query_variant_limit,
        args.frame_neighborhood_count,
        args.vqa_qwen_candidate_budget,
        args.trake_event_top_k,
        args.trake_qwen_per_event,
        args.review_top_k,
    ) < 1:
        raise SystemExit("KIS candidate/review budgets must be positive")
    if args.offline and args.allow_external_search:
        raise SystemExit("--offline and --allow-external-search cannot be used together")
    if args.resume and args.checkpoint_dir is None:
        raise SystemExit("--resume requires --checkpoint-dir")
    if args.provenance_dir is None:
        args.provenance_dir = args.output.with_suffix("").with_name(f"{args.output.stem}_provenance")
    if args.review_output_dir is None:
        # Candidate-level evidence is part of every reproducible run, even
        # when the caller is not yet ready to make manual decisions.
        args.review_output_dir = args.provenance_dir / "kis_review"
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    specs = load_query_specs(args.queries_dir, args.manifest)
    tasks = _build_tasks(args, specs)
    temporary_work = None
    if args.checkpoint_dir is None:
        temporary_work = tempfile.TemporaryDirectory(prefix="aic2026-submission-")
        submission_dir = Path(temporary_work.name) / "submission"
    else:
        submission_dir = args.checkpoint_dir.resolve()
    submission_dir.mkdir(parents=True, exist_ok=True)

    try:
        for spec in specs:
            output_csv = submission_dir / f"{spec.query_id}.csv"
            if args.resume and output_csv.exists():
                checkpoint_rows = read_csv_file(output_csv)
                checkpoint_errors = validate_rows(checkpoint_rows, spec, args.max_rows)
                if not checkpoint_errors:
                    LOGGER.info("%s: resumed %d validated rows", spec.query_id, len(checkpoint_rows))
                    continue
                LOGGER.warning(
                    "%s: checkpoint is invalid and will be regenerated: %s",
                    spec.query_id,
                    "; ".join(checkpoint_errors),
                )

            rows = _generate_for_query(spec, tasks, args)
            if not rows:
                raise RuntimeError(f"Query {spec.query_id} produced no valid submission rows")
            errors = validate_rows([[str(value) for value in row] for row in rows], spec, args.max_rows)
            if errors:
                raise RuntimeError(f"Generated CSV for {spec.query_id} failed validation: {'; '.join(errors)}")
            temporary_csv = output_csv.with_suffix(output_csv.suffix + ".tmp")
            write_csv(temporary_csv, rows)
            temporary_csv.replace(output_csv)
            LOGGER.info("%s: checkpointed %d rows", spec.query_id, len(rows))

        # Package exactly the requested query set, even if a reused checkpoint
        # directory contains CSVs from a different pack.
        with tempfile.TemporaryDirectory(prefix="aic2026-package-") as package_temp:
            package_dir = Path(package_temp) / "submission"
            package_dir.mkdir(parents=True, exist_ok=True)
            for spec in specs:
                shutil.copy2(submission_dir / f"{spec.query_id}.csv", package_dir)
            package_submission(package_dir, args.output)
    finally:
        if temporary_work is not None:
            temporary_work.cleanup()
    LOGGER.info("Created %s", args.output)
    if args.review_output_dir is not None:
        template = write_review_manifest_template(
            args.review_output_dir,
            [spec.query_id for spec in specs],
        )
        LOGGER.info("Wrote editable review template %s", template)
    LOGGER.info("Wrote reproducibility record %s", _write_provenance(args, specs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
