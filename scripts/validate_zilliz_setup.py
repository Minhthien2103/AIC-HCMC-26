#!/usr/bin/env python3
"""Validate Colab paths, Zilliz collections, and canonical frame mapping."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src import config  # noqa: E402
from src.online_pipeline.zilliz_object_filter import ZillizObjectFilter  # noqa: E402
from src.online_pipeline.zilliz_retrieval import ZillizRetrievalEngine  # noqa: E402
from src.submission.query_parser import load_query_specs  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-path", type=Path, required=True)
    parser.add_argument("--keyframes-dir", type=Path, required=True)
    parser.add_argument("--queries-dir", type=Path)
    parser.add_argument("--visual-collection", default=config.ZILLIZ_VISUAL_COLLECTION)
    parser.add_argument("--text-collection", default=config.ZILLIZ_TEXT_COLLECTION)
    parser.add_argument("--object-collection", default=config.ZILLIZ_OBJECT_COLLECTION)
    parser.add_argument("--keyframe-samples", type=int, default=20)
    return parser.parse_args()


def _keyframe_path(root: Path, video_id: str, name: str) -> Path:
    return config.keyframe_path(video_id, name, root=root)


def main() -> int:
    args = _parse_args()
    uri = os.getenv("ZILLIZ_URI", "").strip()
    token = os.getenv("ZILLIZ_TOKEN", "").strip()
    if not uri or not token:
        raise SystemExit(
            "Missing ZILLIZ_URI/ZILLIZ_TOKEN. Add both in Colab Secrets and run the secret-loading cell."
        )

    retriever = ZillizRetrievalEngine(
        uri=uri,
        token=token,
        metadata_path=args.metadata_path,
        visual_collection=args.visual_collection,
        text_collection=args.text_collection,
        vector_field=config.ZILLIZ_VECTOR_FIELD,
        visual_pk_field=config.ZILLIZ_VISUAL_PK_FIELD,
        text_pk_field=config.ZILLIZ_TEXT_PK_FIELD,
        metric_type=config.ZILLIZ_METRIC_TYPE,
        dimension=config.ZILLIZ_CLIP_DIM,
    )
    ZillizObjectFilter(
        retriever.client,
        args.object_collection,
        retriever.meta_df,
        pk_field=config.ZILLIZ_OBJECT_PK_FIELD,
    )

    vector = np.ones(config.ZILLIZ_CLIP_DIM, dtype=np.float32)
    vector /= np.linalg.norm(vector)
    candidates = retriever.search(vector, top_k=3)
    if not candidates:
        raise RuntimeError(
            "Zilliz search returned no frame mapped by metadata.parquet; collections and metadata likely do not match."
        )

    keyframe_root = args.keyframes_dir.expanduser()
    sample_rows = retriever.meta_df.head(max(1, args.keyframe_samples)).to_dicts()
    missing = [
        _keyframe_path(keyframe_root, row["video_id"], row["keyframe_name"])
        for row in sample_rows
        if not _keyframe_path(keyframe_root, row["video_id"], row["keyframe_name"]).is_file()
    ]
    if missing:
        preview = "\n".join(f"  - {path}" for path in missing[:5])
        raise FileNotFoundError(
            f"Keyframe path does not match canonical metadata ({len(missing)}/{len(sample_rows)} samples missing):\n{preview}"
        )

    if args.queries_dir is not None:
        specs = load_query_specs(queries_dir=args.queries_dir)
        type_counts = {
            query_type: sum(spec.query_type == query_type for spec in specs)
            for query_type in ("kis", "qa", "trake")
        }
        empty_trake = [spec.query_id for spec in specs if spec.query_type == "trake" and not spec.events]
        if empty_trake:
            raise RuntimeError(f"TRAKE queries without parsed events: {empty_trake}")
        print(f"Query pack: {len(specs)} files {type_counts}")

    print(f"Metadata: {retriever.meta_df.height:,} canonical frames")
    print(
        "Zilliz: OK "
        f"(visual={args.visual_collection}, text={args.text_collection}, object={args.object_collection})"
    )
    print(f"Mapped smoke result: {candidates[0]['frame_key']}")
    print(f"Keyframes: {len(sample_rows)} sampled paths found")
    print("Setup validation: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
