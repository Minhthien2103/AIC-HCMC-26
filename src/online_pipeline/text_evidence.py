"""Offline text evidence, BTC media metadata retrieval and candidate OCR.

The final submission generator only reads caches and local indices from this
module.  Network access belongs exclusively to ``prepare_kis_assets.py``.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np


class E5TextEncoder:
    """Lazy wrapper around multilingual E5 with explicit query/document tags."""

    def __init__(self, model_name: str, *, device: str = "cuda", local_files_only: bool = True):
        self.model_name = model_name
        self.device = device
        self.local_files_only = local_files_only
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                local_files_only=self.local_files_only,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Unable to load local E5 model {self.model_name}. Run prepare_kis_assets on the GPU runtime."
            ) from exc

    def encode_queries(self, values: list[str]) -> np.ndarray:
        self._load()
        return np.asarray(
            self._model.encode(
                [f"query: {value}" for value in values],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )

    def encode_documents(self, values: list[str], *, batch_size: int = 64) -> np.ndarray:
        self._load()
        return np.asarray(
            self._model.encode(
                [f"passage: {value}" for value in values],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=batch_size,
            ),
            dtype=np.float32,
        )


class MediaTextRetriever:
    """Search an E5 FAISS index containing official BTC media metadata."""

    def __init__(self, index_path: str | Path, records_path: str | Path, encoder: E5TextEncoder):
        self.index_path = Path(index_path)
        self.records_path = Path(records_path)
        self.encoder = encoder
        self.index = None
        self.records: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if not self.index_path.exists() or not self.records_path.exists():
            raise FileNotFoundError(
                "Media E5 assets are missing. Run scripts/prepare_kis_assets.py --build-media-index first."
            )
        try:
            import faiss

            self.index = faiss.read_index(str(self.index_path))
            payload = json.loads(self.records_path.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                raise ValueError("Media E5 records must be a JSON array")
            self.records = [dict(item) for item in payload]
            if self.index.ntotal != len(self.records):
                raise ValueError(f"media E5 index rows={self.index.ntotal}, records={len(self.records)}")
        except Exception as exc:
            raise RuntimeError(f"Could not load media text retrieval assets: {exc}") from exc

    def search(self, query: str, *, top_k: int = 30) -> list[dict[str, Any]]:
        if not query.strip() or self.index is None:
            return []
        vector = self.encoder.encode_queries([query])
        limit = max(0, min(int(top_k), int(self.index.ntotal)))
        if limit == 0:
            return []
        scores, indices = self.index.search(vector, limit)
        # Several metadata rows can describe one video. Preserve its best row.
        by_video: dict[str, dict[str, Any]] = {}
        for rank, (score, index) in enumerate(zip(scores[0], indices[0]), start=1):
            if int(index) < 0:
                continue
            record = self.records[int(index)].copy()
            video_id = str(record.get("video_id", "")).strip()
            if not video_id:
                continue
            candidate = {
                "video_id": video_id,
                "metadata_score": float(score),
                "metadata_rank": rank,
                "metadata_text": str(record.get("text", "")),
                "metadata_record": record,
            }
            if video_id not in by_video:
                by_video[video_id] = candidate
        return list(by_video.values())


class OfflineEvidenceCache:
    """Read-only cache of pre-fetched external evidence for query analysis."""

    def __init__(self, path: str | Path, *, required: bool = False):
        self.path = Path(path)
        self.required = required
        self.payload: dict[str, Any] = {}
        if self.path.exists():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError(f"Evidence cache must be a JSON object: {self.path}")
            self.payload = raw
        elif required:
            raise FileNotFoundError(f"Offline evidence cache is required but missing: {self.path}")

    def context(self, query_id: str) -> str:
        item = self.payload.get(query_id)
        if item is None:
            if self.required:
                raise KeyError(f"Offline evidence cache has no entry for {query_id}")
            return ""
        docs = item.get("documents", []) if isinstance(item, dict) else []
        if not isinstance(docs, list):
            return ""
        pieces: list[str] = []
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            title = str(doc.get("title", "")).strip()
            body = str(doc.get("body", "")).strip()
            if title or body:
                pieces.append(f"{title}: {body}".strip(": "))
        return "\n".join(pieces)


class CandidateOCR:
    """OCR only a retrieval pool and persist keyed text by image content hash."""

    def __init__(
        self,
        cache_dir: str | Path,
        encoder: E5TextEncoder,
        *,
        extractor: Callable[[Path], str] | None = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.encoder = encoder
        self.extractor = extractor
        self._paddle = None

    def _cache_path(self, image_path: Path) -> Path:
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _paddle_extract(self, image_path: Path) -> str:
        if self._paddle is None:
            try:
                from paddleocr import PaddleOCR

                self._paddle = PaddleOCR(lang="en", use_angle_cls=True, show_log=False)
            except TypeError:  # PaddleOCR 3.x removed old constructor options.
                from paddleocr import PaddleOCR

                self._paddle = PaddleOCR(lang="en")
            except Exception as exc:
                raise RuntimeError("PaddleOCR is unavailable; install it during asset preparation.") from exc
        try:
            result = self._paddle.ocr(str(image_path), cls=True)
        except AttributeError:  # PaddleOCR 3.x API.
            result = self._paddle.predict(str(image_path))
        texts: list[str] = []
        # Legacy .ocr output: [[[[coords], (text, confidence)], ...]]
        for page in result or []:
            if isinstance(page, dict):
                values = page.get("rec_texts") or page.get("text") or []
                texts.extend(str(value) for value in values if str(value).strip())
            elif isinstance(page, list):
                for line in page:
                    if isinstance(line, (list, tuple)) and len(line) >= 2:
                        value = line[1]
                        if isinstance(value, (list, tuple)) and value:
                            texts.append(str(value[0]))
        return " ".join(text.strip() for text in texts if text.strip())

    def text(self, image_path: str | Path) -> str:
        resolved = Path(image_path)
        cache_path = self._cache_path(resolved)
        if cache_path.exists():
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
            return str(raw.get("text", ""))
        output = (self.extractor or self._paddle_extract)(resolved)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {"image_sha256": cache_path.stem, "text": str(output), "created_at": datetime.now(timezone.utc).isoformat()},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        temporary.replace(cache_path)
        return str(output)

    def rank(self, query: str, candidates: list[dict[str, Any]], image_path_for: Callable[[dict[str, Any]], Path]) -> list[dict[str, Any]]:
        texts: list[str] = []
        materialized: list[dict[str, Any]] = []
        for candidate in candidates:
            item = candidate.copy()
            item["ocr_text"] = self.text(image_path_for(item))
            if item["ocr_text"].strip():
                texts.append(item["ocr_text"])
                materialized.append(item)
        if not materialized:
            return []
        q_vector = self.encoder.encode_queries([query])[0]
        vectors = self.encoder.encode_documents(texts)
        scores = vectors @ q_vector
        for item, score in zip(materialized, scores):
            item["ocr_score"] = float(score)
            item["score"] = float(score)
        materialized.sort(key=lambda item: float(item["score"]), reverse=True)
        return materialized
