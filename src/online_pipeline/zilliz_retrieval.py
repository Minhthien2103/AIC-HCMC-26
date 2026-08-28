"""MobileCLIP retrieval over the production Zilliz visual/text collections."""

from __future__ import annotations

import json
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from src import config
from src.online_pipeline.rank_fusion import fuse_rankings


@dataclass(frozen=True)
class _IndexInfo:
    """Small compatibility surface used by the existing task budget logic."""

    ntotal: int
    d: int


def load_canonical_metadata(metadata_path: str | Path) -> pl.DataFrame:
    """Load the mapping file and expose the legacy candidate field contract.

    The generated mapping uses ``frame_name``/``frame_idx`` while the online
    tasks use ``keyframe_name``/``frame_id``. ``faiss_idx`` is retained only as
    a stable, local row identity for rank fusion; it is never used to join
    Zilliz collections.
    """

    path = Path(metadata_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"Canonical metadata not found: {path}. Set AIC_METADATA_PATH or pass --metadata-path."
        )
    metadata = pl.read_parquet(path)
    required_source = {"video_id"}
    missing = required_source - set(metadata.columns)
    if missing:
        raise ValueError(f"Canonical metadata is missing columns: {sorted(missing)}")

    if "keyframe_name" not in metadata.columns:
        if "frame_name" not in metadata.columns:
            raise ValueError("Canonical metadata needs frame_name or keyframe_name")
        metadata = metadata.with_columns(pl.col("frame_name").alias("keyframe_name"))
    if "frame_id" not in metadata.columns:
        if "frame_idx" not in metadata.columns:
            raise ValueError("Canonical metadata needs frame_idx or frame_id")
        metadata = metadata.with_columns(pl.col("frame_idx").alias("frame_id"))
    if "frame_name" not in metadata.columns:
        metadata = metadata.with_columns(pl.col("keyframe_name").alias("frame_name"))
    if "frame_idx" not in metadata.columns:
        metadata = metadata.with_columns(pl.col("frame_id").alias("frame_idx"))
    if "timestamp" not in metadata.columns:
        metadata = metadata.with_columns(pl.lit(None, dtype=pl.Float64).alias("timestamp"))
    if "pts_time" not in metadata.columns:
        metadata = metadata.with_columns(pl.col("timestamp").alias("pts_time"))

    metadata = metadata.with_columns(
        pl.col("video_id").cast(pl.String),
        pl.col("frame_name").cast(pl.String),
        pl.col("keyframe_name").cast(pl.String),
        pl.col("frame_idx").cast(pl.Int64),
        pl.col("frame_id").cast(pl.Int64),
        pl.col("timestamp").cast(pl.Float64, strict=False),
        pl.col("pts_time").cast(pl.Float64, strict=False),
    )
    if "frame_key" not in metadata.columns:
        metadata = metadata.with_columns(
            pl.concat_str(
                [pl.col("video_id"), pl.lit("::"), pl.col("frame_idx").cast(pl.String)]
            ).alias("frame_key")
        )
    else:
        metadata = metadata.with_columns(pl.col("frame_key").cast(pl.String))
    if "visual_pk" in metadata.columns:
        metadata = metadata.with_columns(pl.col("visual_pk").cast(pl.String))

    metadata = metadata.sort(["video_id", "frame_id", "keyframe_name"])
    if "faiss_idx" in metadata.columns:
        metadata = metadata.drop("faiss_idx")
    metadata = metadata.with_row_index("faiss_idx")

    duplicates = metadata.group_by("frame_key").len().filter(pl.col("len") > 1)
    if duplicates.height:
        raise ValueError(f"Canonical metadata frame_key is not unique: {duplicates.head(5)}")
    return metadata


class ZillizRetrievalEngine:
    """Search visual and subtitle vectors, then map every hit to one frame."""

    VISUAL_OUTPUT_FIELDS = ["video_id", "frame_name", "frame_idx", "timestamp"]
    TEXT_OUTPUT_FIELDS = [
        "video_id",
        "timestamp_start",
        "timestamp_end",
        "frame_number_start",
        "frame_number_end",
        "subtitles",
    ]

    def __init__(
        self,
        *,
        uri: str,
        token: str,
        metadata_path: str | Path,
        visual_collection: str,
        text_collection: str,
        vector_field: str = "embedding",
        visual_pk_field: str = "pk",
        text_pk_field: str = "pk",
        metric_type: str = "COSINE",
        dimension: int = 512,
        keyframes_dir: str | Path | None = None,
        client=None,
    ):
        self.uri = str(uri or "").strip()
        self.visual_collection = str(visual_collection)
        self.text_collection = str(text_collection)
        self.vector_field = str(vector_field)
        self.visual_pk_field = str(visual_pk_field)
        self.text_pk_field = str(text_pk_field)
        self.metric_type = str(metric_type).upper()
        self.keyframes_dir = (
            Path(keyframes_dir).expanduser() if keyframes_dir is not None else None
        )
        self._local_keyframe_cache: dict[str, bool] = {}
        self.meta_df = load_canonical_metadata(metadata_path)
        self.index = _IndexInfo(ntotal=self.meta_df.height, d=int(dimension))

        if client is None:
            if not self.uri or not str(token or "").strip():
                raise RuntimeError(
                    "Missing ZILLIZ_URI/ZILLIZ_TOKEN. Add them to Colab Secrets and export them as environment variables."
                )
            try:
                from pymilvus import MilvusClient
            except ImportError as exc:
                raise RuntimeError("Zilliz retrieval requires pymilvus; install requirements.txt") from exc
            client = MilvusClient(uri=self.uri, token=str(token))
        self.client = client
        available = set(self.client.list_collections())
        missing_collections = {
            self.visual_collection,
            self.text_collection,
        } - available
        if missing_collections:
            raise RuntimeError(
                "Zilliz is connected but required collections are missing: "
                + ", ".join(sorted(missing_collections))
            )
        if hasattr(self.client, "describe_collection"):
            expected = {
                self.visual_collection: {
                    self.visual_pk_field,
                    self.vector_field,
                    *self.VISUAL_OUTPUT_FIELDS,
                },
                self.text_collection: {
                    self.text_pk_field,
                    self.vector_field,
                    *self.TEXT_OUTPUT_FIELDS,
                },
            }
            for collection_name, required_fields in expected.items():
                description = self.client.describe_collection(collection_name)
                actual_fields = {
                    str(field.get("name")) for field in description.get("fields", [])
                }
                missing_fields = required_fields - actual_fields
                if missing_fields:
                    raise RuntimeError(
                        f"Zilliz collection {collection_name} is missing fields: {sorted(missing_fields)}"
                    )

        rows = self.meta_df.to_dicts()
        self._by_visual_pk: dict[str, dict[str, Any]] = {}
        self._by_text_pk: dict[str, dict[str, Any]] = {}
        self._by_frame_key: dict[str, dict[str, Any]] = {}
        self._by_video_frame: dict[tuple[str, int], dict[str, Any]] = {}
        self._by_video_name: dict[tuple[str, str], dict[str, Any]] = {}
        self._video_rows: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            canonical = dict(row)
            video_id = str(canonical["video_id"])
            frame_id = int(canonical["frame_id"])
            self._by_frame_key[str(canonical["frame_key"])] = canonical
            self._by_video_frame[(video_id, frame_id)] = canonical
            self._by_video_name[(video_id, str(canonical["frame_name"]))] = canonical
            self._by_video_name[(video_id, str(canonical["keyframe_name"]))] = canonical
            if canonical.get("visual_pk") is not None:
                self._by_visual_pk[str(canonical["visual_pk"])] = canonical
            text_pks = canonical.get("text_pks") or []
            if not isinstance(text_pks, (list, tuple)):
                text_pks = [text_pks]
            for text_pk in text_pks:
                if text_pk is not None:
                    self._by_text_pk[str(text_pk)] = canonical
            self._video_rows.setdefault(video_id, []).append(canonical)
        for video_rows in self._video_rows.values():
            video_rows.sort(key=lambda row: int(row["frame_id"]))

    def _has_local_keyframe(self, candidate: dict[str, Any]) -> bool:
        if self.keyframes_dir is None:
            return True
        identity = str(candidate.get("frame_key") or "")
        if identity not in self._local_keyframe_cache:
            path = config.keyframe_path(
                str(candidate["video_id"]),
                str(candidate["keyframe_name"]),
                root=self.keyframes_dir,
            )
            self._local_keyframe_cache[identity] = path.is_file()
        return self._local_keyframe_cache[identity]

    @staticmethod
    def _entity(hit: Any) -> tuple[dict[str, Any], Any, float]:
        value = dict(hit) if not isinstance(hit, dict) else hit
        entity = value.get("entity") or {}
        entity = dict(entity) if not isinstance(entity, dict) else entity
        return entity, value.get("id"), float(value.get("distance", value.get("score", 0.0)))

    def _candidate_for_visual(self, hit: Any) -> dict[str, Any] | None:
        entity, hit_id, score = self._entity(hit)
        primary = entity.get(self.visual_pk_field, hit_id)
        row = self._by_visual_pk.get(str(primary)) if primary is not None else None
        video_id = str(entity.get("video_id", ""))
        frame_value = entity.get("frame_idx")
        if row is None and video_id and frame_value is not None:
            row = self._by_video_frame.get((video_id, int(frame_value)))
        if row is None and video_id and entity.get("frame_name") is not None:
            row = self._by_video_name.get((video_id, str(entity["frame_name"])))
        if row is None:
            return None
        candidate = dict(row)
        candidate.update(
            score=score,
            visual_score=score,
            retrieval_modality="visual",
            zilliz_visual_pk=str(primary) if primary is not None else candidate.get("visual_pk"),
        )
        return candidate

    @staticmethod
    def _midpoint(first: Any, second: Any) -> float | None:
        values = []
        for value in (first, second):
            if value is not None:
                try:
                    values.append(float(value))
                except (TypeError, ValueError):
                    pass
        return sum(values) / len(values) if values else None

    def _nearest_frame(self, entity: dict[str, Any]) -> dict[str, Any] | None:
        video_id = str(entity.get("video_id", ""))
        rows = self._video_rows.get(video_id)
        if not rows:
            return None
        # The canonical metadata notebook anchors subtitle segments by nearest
        # visual timestamp, so use the identical fallback when text_pks is not
        # present in an older mapping file.
        target_time = self._midpoint(entity.get("timestamp_start"), entity.get("timestamp_end"))
        if target_time is not None:
            timed = [row for row in rows if row.get("timestamp") is not None]
            if timed:
                return min(timed, key=lambda row: abs(float(row["timestamp"]) - target_time))
        target_frame = self._midpoint(
            entity.get("frame_number_start"), entity.get("frame_number_end")
        )
        if target_frame is not None:
            frame_ids = [int(row["frame_id"]) for row in rows]
            position = bisect_left(frame_ids, target_frame)
            choices = [index for index in (position - 1, position) if 0 <= index < len(rows)]
            if not choices:
                return None
            nearest = min(choices, key=lambda index: abs(frame_ids[index] - target_frame))
            return rows[nearest]

        return rows[0]

    def _candidate_for_text(self, hit: Any) -> dict[str, Any] | None:
        entity, hit_id, score = self._entity(hit)
        primary = entity.get(self.text_pk_field, hit_id)
        row = self._by_text_pk.get(str(primary)) if primary is not None else None
        if row is None:
            row = self._nearest_frame(entity)
        if row is None:
            return None
        candidate = dict(row)
        candidate.update(
            score=score,
            text_score=score,
            retrieval_modality="subtitle",
            zilliz_text_pk=str(primary) if primary is not None else None,
            subtitle=str(entity.get("subtitles") or ""),
            subtitle_timestamp_start=entity.get("timestamp_start"),
            subtitle_timestamp_end=entity.get("timestamp_end"),
            subtitle_frame_start=entity.get("frame_number_start"),
            subtitle_frame_end=entity.get("frame_number_end"),
        )
        return candidate

    def _search_collection(
        self,
        collection_name: str,
        vectors: np.ndarray,
        *,
        limit: int,
        output_fields: list[str],
        filter_expression: str,
    ) -> list[list[dict[str, Any]]]:
        fields = list(dict.fromkeys(output_fields))
        return self.client.search(
            collection_name=collection_name,
            data=vectors.tolist(),
            anns_field=self.vector_field,
            filter=filter_expression,
            limit=limit,
            output_fields=fields,
            search_params={"metric_type": self.metric_type, "params": {}},
        )

    def search(self, query_vector: np.ndarray, top_k: int = 100) -> list[dict[str, Any]]:
        rows = self.search_batch(query_vector, top_k=top_k)
        return rows[0] if rows else []

    def search_batch(self, query_vectors: np.ndarray, top_k: int = 100) -> list[list[dict[str, Any]]]:
        return self._search_batch(query_vectors, top_k=top_k, video_id=None)

    def search_in_video(self, query_vector: np.ndarray, video_id: str, top_k: int = 5) -> list[dict[str, Any]]:
        rows = self.search_in_video_batch(query_vector, video_id=video_id, top_k=top_k)
        return rows[0] if rows else []

    def search_in_video_batch(
        self, query_vectors: np.ndarray, video_id: str, top_k: int = 5
    ) -> list[list[dict[str, Any]]]:
        return self._search_batch(query_vectors, top_k=top_k, video_id=str(video_id))

    def _search_batch(
        self, query_vectors: np.ndarray, *, top_k: int, video_id: str | None
    ) -> list[list[dict[str, Any]]]:
        vectors = np.asarray(query_vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.ndim != 2 or vectors.shape[1] != self.index.d:
            raise ValueError(f"Query vectors must have shape (N, {self.index.d}), got {vectors.shape}")
        if vectors.shape[0] == 0:
            return []
        requested_limit = min(max(0, int(top_k)), self.index.ntotal)
        if requested_limit == 0:
            return [[] for _ in range(vectors.shape[0])]
        search_limit = requested_limit
        if self.keyframes_dir is not None:
            search_limit = min(
                self.index.ntotal,
                max(requested_limit * 3, requested_limit + 32),
            )
        expression = f"video_id == {json.dumps(video_id)}" if video_id is not None else ""
        visual_raw = self._search_collection(
            self.visual_collection,
            vectors,
            limit=search_limit,
            output_fields=[self.visual_pk_field, *self.VISUAL_OUTPUT_FIELDS],
            filter_expression=expression,
        )
        text_raw = self._search_collection(
            self.text_collection,
            vectors,
            limit=search_limit,
            output_fields=[self.text_pk_field, *self.TEXT_OUTPUT_FIELDS],
            filter_expression=expression,
        )
        output: list[list[dict[str, Any]]] = []
        for index in range(vectors.shape[0]):
            visual = [
                candidate
                for hit in (visual_raw[index] if index < len(visual_raw) else [])
                if (candidate := self._candidate_for_visual(hit)) is not None
            ]
            text = [
                candidate
                for hit in (text_raw[index] if index < len(text_raw) else [])
                if (candidate := self._candidate_for_text(hit)) is not None
            ]
            fused = fuse_rankings(
                [("zilliz_visual", visual), ("zilliz_subtitle", text)],
                rrf_k=60,
            )
            available = []
            for candidate in fused:
                if self._has_local_keyframe(candidate):
                    available.append(candidate)
                    if len(available) >= requested_limit:
                        break
            output.append(available)
        return output
