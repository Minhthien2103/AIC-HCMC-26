"""Object-evidence lookup for candidates stored in Zilliz Cloud."""

from __future__ import annotations

import json
from typing import Any

import polars as pl

from src import config


class ZillizObjectFilter:
    def __init__(
        self,
        client,
        collection_name: str,
        metadata: pl.DataFrame,
        *,
        pk_field: str = "pk",
        query_batch_size: int = 100,
    ):
        self.client = client
        self.collection_name = str(collection_name)
        self.pk_field = str(pk_field)
        self.query_batch_size = max(1, int(query_batch_size))
        if self.collection_name not in set(self.client.list_collections()):
            raise RuntimeError(f"Zilliz object collection is missing: {self.collection_name}")
        if hasattr(self.client, "describe_collection"):
            description = self.client.describe_collection(self.collection_name)
            fields = {str(field.get("name")) for field in description.get("fields", [])}
            missing = {self.pk_field, "frame_key", "class_name"} - fields
            if missing:
                raise RuntimeError(
                    f"Zilliz object collection {self.collection_name} is missing fields: {sorted(missing)}"
                )

        labels: set[str] = set()
        if "object_class_names" in metadata.columns:
            for value in metadata["object_class_names"].drop_nulls().to_list():
                if isinstance(value, (list, tuple)):
                    labels.update(str(item) for item in value if str(item).strip())
                elif str(value).strip():
                    labels.add(str(value))
        self._all_labels = sorted(labels, key=str.casefold)

    def get_all_labels(self) -> list[str]:
        return list(self._all_labels)

    @staticmethod
    def _frame_key(candidate: dict[str, Any]) -> str:
        if candidate.get("frame_key") is not None:
            return str(candidate["frame_key"])
        frame_id = candidate.get("frame_idx", candidate.get("frame_id"))
        return f"{candidate.get('video_id', '')}::{frame_id}"

    def _detected_labels(self, candidates: list[dict[str, Any]]) -> dict[str, set[str]]:
        keys = list(dict.fromkeys(self._frame_key(candidate) for candidate in candidates))
        lookup: dict[str, set[str]] = {}
        for start in range(0, len(keys), self.query_batch_size):
            batch = keys[start : start + self.query_batch_size]
            encoded = ", ".join(json.dumps(value) for value in batch)
            rows = self.client.query(
                collection_name=self.collection_name,
                filter=f"frame_key in [{encoded}]",
                output_fields=["frame_key", "class_name"],
                limit=16384,
            )
            for row in rows:
                key = str(row.get("frame_key", ""))
                label = str(row.get("class_name", "")).strip().casefold()
                if key and label:
                    lookup.setdefault(key, set()).add(label)
        return lookup

    def matching_candidates(
        self,
        candidates: list[dict[str, Any]],
        required_objects: list[str],
        *,
        require_all: bool = False,
    ) -> list[dict[str, Any]]:
        required = {value.strip().casefold() for value in required_objects if value.strip()}
        if not candidates or not required:
            return []
        lookup = self._detected_labels(candidates)
        output = []
        for candidate in candidates:
            found = lookup.get(self._frame_key(candidate), set())
            overlap = required.intersection(found)
            if (overlap == required if require_all else bool(overlap)):
                item = candidate.copy()
                item["matched_objects"] = sorted(overlap)
                output.append(item)
        return output

    def filter_candidates(
        self,
        candidates: list[dict[str, Any]],
        required_objects: list[str],
        mode: str = "boost",
    ) -> list[dict[str, Any]]:
        if not candidates or not required_objects:
            return candidates
        required = {value.strip().casefold() for value in required_objects if value.strip()}
        lookup = self._detected_labels(candidates)
        output = []
        for candidate in candidates:
            found = lookup.get(self._frame_key(candidate), set())
            overlap = required.intersection(found)
            if mode == "hard":
                if overlap == required:
                    item = candidate.copy()
                    item["matched_objects"] = sorted(overlap)
                    output.append(item)
                continue
            item = candidate.copy()
            item["clip_score"] = float(item.get("clip_score", item.get("score", 0.0)))
            item["score"] = (
                item["clip_score"] * config.CLIP_WEIGHT
                + (len(overlap) / len(required)) * config.OBJECT_BOOST_WEIGHT
            )
            item["matched_objects"] = sorted(overlap)
            output.append(item)
        output.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        return output
