#!/usr/bin/env python3
"""Prepare resumable local assets for the offline KIS final run.

Run this on a CUDA Colab runtime before generating a final submission. It may
use the network only while explicitly downloading BTC metadata or building the
evidence cache; ``generate_submission.py --offline`` never performs requests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import polars as pl
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from src import config  # noqa: E402
from src.online_pipeline.query_encoder import QueryEncoder  # noqa: E402
from src.online_pipeline.retrieval import RetrievalEngine  # noqa: E402
from src.online_pipeline.text_evidence import CandidateOCR, MediaTextRetriever  # noqa: E402
from src.online_pipeline.text_evidence import E5TextEncoder  # noqa: E402
from src.online_pipeline.vlm_pipeline import VLMPipeline  # noqa: E402
from src.submission.query_parser import load_query_specs  # noqa: E402
from src.utils.translation import configure_translation, translate_vi_to_en  # noqa: E402


LOGGER = logging.getLogger("aic2026.prepare_kis_assets")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--download-media-info", action="store_true")
    parser.add_argument("--media-info-url", default=config.MEDIA_INFO_ARCHIVE_URL)
    parser.add_argument("--media-info-archive", type=Path)
    parser.add_argument("--build-media-index", action="store_true")
    parser.add_argument("--build-vith-index", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--checkpoint-every", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--build-evidence-cache", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="Check real dual indexes and OCR cache on one keyframe.")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "query" / "manifest_full.json")
    parser.add_argument("--evidence-cache", type=Path)
    parser.add_argument("--evidence-results-per-query", type=int, default=5)
    parser.add_argument("--translation-cache", type=Path)
    return parser.parse_args()


def _configure_paths(repo_root: Path) -> None:
    config.BASE_DIR = repo_root.resolve()
    config.DATA_DIR = config.BASE_DIR / "data"
    config.INDEX_DIR = config.BASE_DIR / "indexes"
    config.KEYFRAMES_DIR = config.DATA_DIR / "keyframes"
    config.MEDIA_INFO_DIR = config.DATA_DIR / "media-info"
    config.METADATA_PATH = config.INDEX_DIR / "metadata.parquet"
    config.VITH_INDEX_PATH = config.INDEX_DIR / "faiss_vith.index"
    config.VITH_FEATURES_PATH = config.INDEX_DIR / "vith_features.f16.npy"
    config.MEDIA_TEXT_INDEX_PATH = config.INDEX_DIR / "media_e5.index"
    config.MEDIA_TEXT_RECORDS_PATH = config.INDEX_DIR / "media_e5_records.json"


def _download_media_info(url: str, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Downloading official media-info archive from %s", url)
    with urllib.request.urlopen(url) as response, archive_path.open("wb") as output:
        shutil.copyfileobj(response, output)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(config.DATA_DIR)
    LOGGER.info("Extracted media metadata to %s", config.MEDIA_INFO_DIR)


def _metadata_records() -> list[dict[str, str]]:
    if not config.MEDIA_INFO_DIR.exists():
        raise FileNotFoundError(f"BTC media metadata missing: {config.MEDIA_INFO_DIR}")
    records: list[dict[str, str]] = []
    for item in sorted(config.MEDIA_INFO_DIR.glob("*.json")):
        try:
            payload = json.loads(item.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid media JSON: {item}") from exc
        if not isinstance(payload, dict):
            continue
        # Keep official raw text instead of a hand-written keyword rule.
        fields = [
            str(payload.get(name, "")).strip()
            for name in ("title", "description", "author", "channel", "tags", "categories")
        ]
        text = "\n".join(value for value in fields if value)
        if text:
            records.append({"video_id": item.stem, "text": text, "metadata_path": str(item)})
    if not records:
        raise RuntimeError(f"No usable JSON records found in {config.MEDIA_INFO_DIR}")
    return records


def build_media_index(device: str) -> None:
    records = _metadata_records()
    encoder = E5TextEncoder(config.E5_MODEL_NAME, device=device, local_files_only=True)
    vectors = encoder.encode_documents([record["text"] for record in records])
    index = faiss.IndexFlatIP(int(vectors.shape[1]))
    index.add(np.ascontiguousarray(vectors, dtype=np.float32))
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(config.MEDIA_TEXT_INDEX_PATH))
    config.MEDIA_TEXT_RECORDS_PATH.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Created media E5 index: %d metadata records, dimension %d", index.ntotal, index.d)


def _metadata_fingerprint(metadata: pl.DataFrame) -> str:
    columns = [name for name in ("faiss_idx", "video_id", "keyframe_name") if name in metadata.columns]
    content = metadata.select(columns).write_csv().encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _vith_state_path() -> Path:
    return config.VITH_FEATURES_PATH.with_suffix(".state.json")


def _load_vith_state(fingerprint: str, *, resume: bool) -> int:
    state_path = _vith_state_path()
    if not resume or not state_path.exists():
        return 0
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("metadata_sha256") != fingerprint:
        raise RuntimeError("ViT-H resume state belongs to different metadata; remove stale assets and rebuild.")
    return int(state.get("next_idx", 0))


def _save_vith_state(next_idx: int, fingerprint: str, dimension: int) -> None:
    state_path = _vith_state_path()
    temporary = state_path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {
                "next_idx": next_idx,
                "metadata_sha256": fingerprint,
                "dimension": dimension,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary.replace(state_path)


def _load_images(rows: list[dict[str, Any]]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for row in rows:
        image_path = config.keyframe_path(str(row["video_id"]), str(row["keyframe_name"]))
        if not image_path.exists():
            raise FileNotFoundError(f"Missing keyframe referenced by metadata: {image_path}")
        with Image.open(image_path) as opened:
            images.append(opened.convert("RGB"))
    return images


def build_vith_index(device: str, batch_size: int, checkpoint_every: int, resume: bool) -> None:
    if batch_size < 1 or checkpoint_every < 1:
        raise ValueError("--batch-size and --checkpoint-every must be positive")
    if not config.METADATA_PATH.exists():
        raise FileNotFoundError(f"Baseline metadata is missing: {config.METADATA_PATH}")
    metadata = pl.read_parquet(config.METADATA_PATH).sort("faiss_idx")
    expected = list(range(metadata.height))
    if metadata["faiss_idx"].cast(pl.Int64).to_list() != expected:
        raise ValueError("metadata.parquet faiss_idx must be contiguous before ViT-H indexing")
    rows = metadata.select(["faiss_idx", "video_id", "keyframe_name"]).to_dicts()
    fingerprint = _metadata_fingerprint(metadata)
    encoder = QueryEncoder(config.VITH_MODEL_NAME, config.VITH_PRETRAINED, device=device)
    config.VITH_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)

    if config.VITH_FEATURES_PATH.exists() and resume:
        features = np.load(config.VITH_FEATURES_PATH, mmap_mode="r+")
        if features.shape[0] != len(rows):
            raise RuntimeError("Existing ViT-H features have the wrong row count; rebuild without --resume.")
        start = _load_vith_state(fingerprint, resume=True)
        dimension = int(features.shape[1])
    else:
        probe = encoder.encode_image_batch(_load_images(rows[:1]))
        dimension = int(probe.shape[1])
        features = np.lib.format.open_memmap(
            config.VITH_FEATURES_PATH, mode="w+", dtype=np.float16, shape=(len(rows), dimension)
        )
        start = 0
        _save_vith_state(start, fingerprint, dimension)
    if not 0 <= start <= len(rows):
        raise RuntimeError(f"Invalid ViT-H resume offset {start}")

    LOGGER.info("Embedding ViT-H keyframes %d/%d (batch=%d)", start, len(rows), batch_size)
    for batch_number, begin in enumerate(range(start, len(rows), batch_size), start=1):
        end = min(begin + batch_size, len(rows))
        vectors = encoder.encode_image_batch(_load_images(rows[begin:end]))
        if vectors.shape != (end - begin, dimension):
            raise RuntimeError(f"Unexpected ViT-H batch shape {vectors.shape}; expected {(end - begin, dimension)}")
        features[begin:end] = vectors.astype(np.float16)
        if batch_number % checkpoint_every == 0 or end == len(rows):
            features.flush()
            _save_vith_state(end, fingerprint, dimension)
            LOGGER.info("ViT-H checkpoint %d/%d", end, len(rows))

    index = faiss.IndexFlatIP(dimension)
    for begin in range(0, len(rows), 8192):
        index.add(np.ascontiguousarray(np.asarray(features[begin:begin + 8192], dtype=np.float32)))
    faiss.write_index(index, str(config.VITH_INDEX_PATH))
    _save_vith_state(len(rows), fingerprint, dimension)
    LOGGER.info("Created ViT-H index: %s (%d vectors, d=%d)", config.VITH_INDEX_PATH, index.ntotal, index.d)


def _search_documents(query: str, count: int) -> list[dict[str, str]]:
    try:
        from duckduckgo_search import DDGS
    except ImportError as exc:
        raise RuntimeError("Evidence preparation requires duckduckgo-search.") from exc
    documents: list[dict[str, str]] = []
    with DDGS() as search:
        for result in search.text(query, max_results=count):
            documents.append(
                {
                    "url": str(result.get("href", "")),
                    "title": str(result.get("title", "")),
                    "body": str(result.get("body", "")),
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                }
            )
    return documents


def build_evidence_cache(args: argparse.Namespace) -> None:
    if args.device != "cuda":
        raise RuntimeError("Evidence planning uses Qwen2-VL and requires --device cuda.")
    cache_path = args.evidence_cache or (config.INDEX_DIR / "kis_evidence_cache.json")
    translation_cache = args.translation_cache or (config.INDEX_DIR / "translation_mbart_cache.json")
    configure_translation(cache_path=translation_cache, offline=False, device="cpu")
    vlm = VLMPipeline(device=args.device)
    payload: dict[str, Any] = {}
    if cache_path.exists():
        existing = json.loads(cache_path.read_text(encoding="utf-8"))
        if isinstance(existing, dict):
            payload = existing
    for spec in load_query_specs(manifest=args.manifest):
        if spec.query_type != "kis" or spec.query_id in payload:
            continue
        english = translate_vi_to_en(spec.description)
        analysis = vlm.analyze_kis_query(english)
        queries = list(dict.fromkeys([english, *analysis.get("retrieval_queries", []), *analysis.get("factual_entities", [])]))
        documents: list[dict[str, str]] = []
        for query in queries[:4]:
            try:
                documents.extend(_search_documents(query, args.evidence_results_per_query))
            except Exception as exc:
                LOGGER.warning("Evidence search failed for %s: %s", spec.query_id, exc)
        unique_docs = {document.get("url") or f"{document.get('title')}:{document.get('body')}": document for document in documents}
        payload[spec.query_id] = {
            "query_sha256": hashlib.sha256(spec.description.encode("utf-8")).hexdigest(),
            "english_query": english,
            "search_queries": queries,
            "documents": list(unique_docs.values()),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        LOGGER.info("Cached offline evidence for %s (%d docs)", spec.query_id, len(unique_docs))


def smoke_test(device: str, manifest: Path) -> None:
    """Exercise every final KIS source using the actual prepared assets."""
    specs = load_query_specs(manifest=manifest)
    query = next((spec.description for spec in specs if spec.query_type == "kis"), "")
    if not query:
        raise RuntimeError("Smoke test needs at least one KIS query in the manifest")
    baseline_encoder = QueryEncoder(config.CLIP_MODEL_NAME, config.CLIP_PRETRAINED, device=device)
    baseline = RetrievalEngine(config.BASE_DIR / "indexes" / "faiss_clip.index", config.METADATA_PATH)
    vith_encoder = QueryEncoder(config.VITH_MODEL_NAME, config.VITH_PRETRAINED, device=device)
    vith = RetrievalEngine(config.VITH_INDEX_PATH, config.METADATA_PATH)
    if baseline.index is None or vith.index is None:
        raise RuntimeError("Smoke test requires both baseline and ViT-H indices")
    e5 = E5TextEncoder(config.E5_MODEL_NAME, device=device, local_files_only=True)
    media = MediaTextRetriever(config.MEDIA_TEXT_INDEX_PATH, config.MEDIA_TEXT_RECORDS_PATH, e5)
    clip_rows = baseline.search(baseline_encoder.encode_text(query), top_k=5)
    vith_rows = vith.search(vith_encoder.encode_text(query), top_k=5)
    media_rows = media.search(query, top_k=5)
    if not clip_rows or not vith_rows or not media_rows:
        raise RuntimeError("Dual retrieval smoke test returned an empty source")
    ocr = CandidateOCR(config.INDEX_DIR / "ocr_cache", e5)
    image_path = config.keyframe_path(clip_rows[0]["video_id"], clip_rows[0]["keyframe_name"])
    first_text = ocr.text(image_path)
    cache_path = ocr._cache_path(image_path)
    if not cache_path.exists():
        raise RuntimeError("OCR cache miss did not persist an entry")
    second_text = ocr.text(image_path)
    if first_text != second_text:
        raise RuntimeError("OCR cache hit changed the extracted text")
    LOGGER.info(
        "KIS smoke passed: ViT-B=%s, ViT-H=%s, media=%s, OCR cache=%s",
        clip_rows[0]["video_id"], vith_rows[0]["video_id"], media_rows[0]["video_id"], cache_path.name,
    )


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    _configure_paths(args.repo_root)
    archive_path = args.media_info_archive or (config.DATA_DIR / "media-info-aic25-b1.zip")
    if args.download_media_info:
        _download_media_info(args.media_info_url, archive_path)
    if args.build_media_index:
        build_media_index(args.device)
    if args.build_vith_index:
        build_vith_index(args.device, args.batch_size, args.checkpoint_every, args.resume)
    if args.build_evidence_cache:
        build_evidence_cache(args)
    if args.smoke_test:
        smoke_test(args.device, args.manifest)
    if not any((args.download_media_info, args.build_media_index, args.build_vith_index, args.build_evidence_cache, args.smoke_test)):
        raise SystemExit("Select at least one preparation action; see --help.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
