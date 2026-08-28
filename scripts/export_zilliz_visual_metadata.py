#!/usr/bin/env python3
"""Export canonical frame metadata from a Zilliz visual collection.

Only scalar fields are requested. The embedding vector is intentionally not
downloaded, so the export is small enough to rebuild at the start of a Kaggle
notebook instead of copying metadata through Google Drive.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Iterable

import polars as pl


DEFAULT_COLLECTION = "aic_visual_mobileclip_s2_v1"
DEFAULT_PK_FIELD = "pk"
SCALAR_FIELDS = ("video_id", "frame_name", "frame_idx", "timestamp")


def iter_collection_scalars(
    client: Any,
    *,
    collection_name: str,
    output_fields: list[str],
    batch_size: int = 2_000,
) -> Iterable[dict[str, Any]]:
    """Yield every scalar entity without Milvus' fixed pagination window."""
    iterator = client.query_iterator(
        collection_name=collection_name,
        batch_size=int(batch_size),
        filter="",
        output_fields=output_fields,
    )
    try:
        while True:
            batch = iterator.next()
            if not batch:
                break
            yield from (dict(row) for row in batch)
    finally:
        iterator.close()


def canonical_metadata(rows: Iterable[dict[str, Any]], *, pk_field: str) -> pl.DataFrame:
    """Normalize visual scalar rows to the online pipeline's mapping schema."""
    records = list(rows)
    if not records:
        raise RuntimeError("The visual collection returned no scalar rows")

    frame = pl.from_dicts(records)
    required = {pk_field, *SCALAR_FIELDS}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Visual scalar export is missing fields: {sorted(missing)}")

    frame = (
        frame.rename({pk_field: "visual_pk"})
        .with_columns(
            pl.col("visual_pk").cast(pl.String),
            pl.col("video_id").cast(pl.String),
            pl.col("frame_name").cast(pl.String),
            pl.col("frame_idx").cast(pl.Int64),
            pl.col("timestamp").cast(pl.Float64, strict=False),
        )
        .with_columns(
            pl.concat_str(
                [pl.col("video_id"), pl.lit("::"), pl.col("frame_idx").cast(pl.String)]
            ).alias("frame_key"),
            pl.concat_str([pl.col("video_id"), pl.lit("/"), pl.col("frame_name")]).alias(
                "image_relative_path"
            ),
        )
        .sort(["video_id", "frame_idx", "frame_name"])
        .with_columns(
            (
                pl.col("frame_idx").rank(method="ordinal").over("video_id").cast(pl.Int64)
                - 1
            ).alias("frame_order")
        )
    )

    nulls = frame.select(required - {pk_field} | {"visual_pk"}).null_count()
    if any(int(value) > 0 for value in nulls.row(0)):
        raise RuntimeError(f"Canonical visual metadata contains null values:\n{nulls}")

    for label, column in (("visual primary key", "visual_pk"), ("frame key", "frame_key")):
        duplicates = frame.group_by(column).len().filter(pl.col("len") > 1)
        if duplicates.height:
            raise RuntimeError(f"Canonical {label} is not unique:\n{duplicates.head(10)}")
    return frame


def export_visual_metadata(
    client: Any,
    *,
    collection_name: str,
    pk_field: str,
    output_path: str | Path,
    batch_size: int = 2_000,
) -> pl.DataFrame:
    """Fetch, validate, and persist one row per canonical visual frame."""
    available = set(client.list_collections())
    if collection_name not in available:
        raise RuntimeError(
            f"Zilliz collection {collection_name!r} is missing; available={sorted(available)}"
        )

    if hasattr(client, "describe_collection"):
        description = client.describe_collection(collection_name)
        actual = {str(field.get("name")) for field in description.get("fields", [])}
        required = {pk_field, *SCALAR_FIELDS}
        missing = required - actual
        if missing:
            raise RuntimeError(
                f"Zilliz collection {collection_name!r} is missing scalar fields: {sorted(missing)}"
            )

    if hasattr(client, "load_collection"):
        client.load_collection(collection_name)
    rows = iter_collection_scalars(
        client,
        collection_name=collection_name,
        output_fields=[pk_field, *SCALAR_FIELDS],
        batch_size=batch_size,
    )
    metadata = canonical_metadata(rows, pk_field=pk_field)
    destination = Path(output_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_parquet(destination)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--collection",
        default=os.getenv("ZILLIZ_VISUAL_COLLECTION", DEFAULT_COLLECTION),
    )
    parser.add_argument(
        "--pk-field",
        default=os.getenv("ZILLIZ_VISUAL_PK_FIELD", DEFAULT_PK_FIELD),
    )
    parser.add_argument("--batch-size", type=int, default=2_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    uri = os.getenv("ZILLIZ_URI", "").strip()
    token = os.getenv("ZILLIZ_TOKEN", "").strip()
    if not uri or not token:
        raise SystemExit("Missing ZILLIZ_URI/ZILLIZ_TOKEN environment variables")

    from pymilvus import MilvusClient

    client = MilvusClient(uri=uri, token=token)
    metadata = export_visual_metadata(
        client,
        collection_name=args.collection,
        pk_field=args.pk_field,
        output_path=args.output,
        batch_size=args.batch_size,
    )
    print(f"Metadata: {metadata.height:,} canonical frames across {metadata['video_id'].n_unique()} videos")
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
