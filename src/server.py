"""FastAPI Server for AI Challenge HCMC 2026 Retrieval System.

Provides high-performance REST and WebSocket endpoints for:
- Text search (single and temporal sequential queries) using FAISS & CLIP
- Visual similarity search (via reconstructed vector indexing)
- Mixed-modal search (text + image)
- Object-filtered search
- Video group search
- Live WebSocket streaming interfaces matching the Baseline Frontend
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

# Add repository root to pythonpath
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src import config
from src.online_pipeline.score_fusion import weighted_rrf, normalized_score
from src.online_pipeline.query_encoder import QueryEncoder
from src.online_pipeline.query_expander import QueryExpander
from src.online_pipeline.retrieval import RetrievalEngine
from src.online_pipeline.text_search import TextSearchEngine
from src.online_pipeline.vlm_reranker import VLMReranker
from src.tasks.kis_t import KIStask
from src.tasks.trake import TrakeTask
from src.utils.translation import translate_vi_to_en

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("aic_server")


# ==============================================================================
# Pydantic Request Models
# ==============================================================================

class TextQueryRequest(BaseModel):
    First_query: str
    Next_query: Optional[str] = ""
    vlm_rerank: Optional[bool] = False


class RerankRequest(BaseModel):
    query: str
    candidates: List[Dict[str, Any]]
    top_n: Optional[int] = 20


class SimilaritySearchRequest(BaseModel):
    vector: Any  # Accepts int faiss_idx or list of floats


class MixedQueryRequest(BaseModel):
    image: Optional[str] = None  # Base64 encoded image
    text: Optional[str] = ""
    ocr_text: str = ""


class DoubleImageQueryRequest(BaseModel):
    firstImage: Optional[str] = None
    secondImage: Optional[str] = None


class VQAQueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 50


def resolve_llm_model(model_name_or_alias: Optional[str]) -> str:
    """Resolve shortcut aliases ('3b', '4b') to full HuggingFace model IDs."""
    if not model_name_or_alias:
        return getattr(config, "LLM_MODEL_ID", "Qwen/Qwen3-4B")
    alias = str(model_name_or_alias).strip().lower()
    if alias in ("3b", "3", "qwen-3b", "qwen2.5-3b", "qwen2.5-3b-instruct"):
        return "Qwen/Qwen2.5-3B-Instruct"
    elif alias in ("4b", "4", "qwen-4b", "qwen3-4b"):
        return "Qwen/Qwen3-4B"
    return str(model_name_or_alias).strip()


def resolve_vlm_model(model_name_or_alias: Optional[str]) -> str:
    """Resolve shortcut aliases ('2b', '3b', '7b') to full HuggingFace VLM model IDs."""
    if not model_name_or_alias:
        return getattr(config, "VLM_MODEL_ID", "Qwen/Qwen2.5-VL-3B-Instruct")
    alias = str(model_name_or_alias).strip().lower()
    if alias in ("2b", "2", "qwen2-vl-2b", "qwen2-vl-2b-instruct"):
        return "Qwen/Qwen2-VL-2B-Instruct"
    elif alias in ("3b", "3", "qwen2.5-vl-3b", "qwen2.5-vl-3b-instruct", "qwen3-vl-2b"):
        return "Qwen/Qwen2.5-VL-3B-Instruct"
    elif alias in ("7b", "7", "qwen2-vl-7b", "qwen2-vl-7b-instruct"):
        return "Qwen/Qwen2-VL-7B-Instruct"
    return str(model_name_or_alias).strip()


# ==============================================================================
# Service Container
# ==============================================================================

class SearchService:
    def __init__(self, llm_model: Optional[str] = None, vlm_model: Optional[str] = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        logger.info(f"Loading CLIP model: {config.CLIP_MODEL_NAME} ({config.CLIP_PRETRAINED})...")
        self.encoder = QueryEncoder(
            model_name=config.CLIP_MODEL_NAME,
            pretrained=config.CLIP_PRETRAINED,
            device=self.device
        )

        logger.info(f"Loading FAISS Index ({config.FAISS_INDEX_PATH}) and Metadata ({config.METADATA_PATH})...")
        self.retriever = RetrievalEngine(
            index_path=config.FAISS_INDEX_PATH,
            metadata_path=config.METADATA_PATH
        )

        logger.info("Initializing KIStask and TrakeTask (Object filter disabled - full CLIP semantic search)...")
        self.object_filter = None
        self.kis_task = KIStask(
            encoder=self.encoder,
            retriever=self.retriever,
            object_filter=None
        )
        self.trake_task = TrakeTask(
            encoder=self.encoder,
            retriever=self.retriever
        )

        # Initialize Local LLM Query Expander (Loads ONLY the chosen model: 3B or 4B)
        if getattr(config, "ENABLE_LLM_EXPANSION", True):
            try:
                selected_llm = resolve_llm_model(llm_model or getattr(config, "LLM_MODEL_ID", "Qwen/Qwen3-4B"))
                llm_device = getattr(config, "LLM_DEVICE", "cuda")
                logger.info(f"Using Query Expander SLM: {selected_llm} on {llm_device}")
                self.expander = QueryExpander(
                    model_id=selected_llm,
                    device=llm_device,
                    load_in_4bit=(llm_device == "cuda")
                )
            except Exception as e:
                logger.warning(f"Could not initialize QueryExpander: {e}")
                self.expander = None
        else:
            self.expander = None

        # Initialize OCR Text Search Engine
        if getattr(config, "ENABLE_OCR_SEARCH", True):
            try:
                self.text_search = TextSearchEngine(ocr_parquet_path=config.OCR_PARQUET_PATH)
            except Exception as e:
                logger.warning(f"Could not initialize TextSearchEngine: {e}")
                self.text_search = None
        else:
            self.text_search = None

        # Initialize VLM Reranker (Single model selected by user or config)
        chosen_vlm = resolve_vlm_model(vlm_model or getattr(config, "VLM_MODEL_ID", "Qwen/Qwen2.5-VL-3B-Instruct"))
        try:
            self.vlm_reranker = VLMReranker(
                model_id=chosen_vlm,
                device=self.device,
                media_base="http://localhost:8007",
                load_mode="4bit"
            )
            logger.info(f"Using VLM Reranker: {chosen_vlm}")
        except Exception as e:
            logger.warning(f"Could not initialize VLMReranker: {e}")
            self.vlm_reranker = None

        # Cache of latest results for pagination
        self.last_results: List[Dict[str, Any]] = []

        logger.info("Service initialized successfully.")

    def format_results(self, candidates: List[Dict[str, Any]], limit: int = 1000) -> List[Dict[str, Any]]:
        """Format candidate dictionary to Milvus-compatible JSON structure expected by frontend."""
        formatted = []
        for idx, cand in enumerate(candidates[:limit]):
            v_id = str(cand.get("video_id") or cand.get("video", ""))
            k_name = str(cand.get("keyframe_name") or cand.get("frame_id", ""))
            pts_time = float(cand.get("pts_time", 0.0))
            score = normalized_score(cand)
            f_idx = int(cand.get("faiss_idx", idx))

            entity_payload: Dict[str, Any] = {
                "video": v_id,
                "frame_id": k_name,
                "time": pts_time,
                "path": f"keyframes/{v_id}/{k_name}.jpg"
            }
            if "ocr_match" in cand:
                entity_payload["ocr_match"] = cand["ocr_match"]
            if "vlm_score" in cand:
                entity_payload["vlm_score"] = cand["vlm_score"]
            if "vlm_verified" in cand:
                entity_payload["vlm_verified"] = cand["vlm_verified"]

            formatted.append({
                "id": f_idx,
                "distance": score,
                "score": score,
                "score_space": "unit",
                "entity": entity_payload
            })
        return formatted

    def decode_base64_image(self, data_url_or_b64: str) -> Image.Image:
        """Decode base64 image data URL to PIL Image."""
        if "," in data_url_or_b64:
            data_url_or_b64 = data_url_or_b64.split(",", 1)[1]
        img_bytes = base64.b64decode(data_url_or_b64)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")

    def _normalize_query_language(self, text: str) -> str:
        """Transparently translate Vietnamese query to English for SigLIP embedding."""
        if not text or not text.strip():
            return ""
        try:
            translated = translate_vi_to_en(text)
            if translated and translated.strip():
                return translated.strip()
        except Exception as e:
            logger.warning(f"Translation fallback for '{text}': {e}")
        return text.strip()

    def encode_query_variants(self, query: str) -> Tuple[np.ndarray, List[str]]:
        """Encode the faithful primary translation and its supporting variants in one batch."""
        clean_q = query.strip()
        if not clean_q:
            return np.empty((1, config.CLIP_DIM), dtype=np.float32), []

        variations = []
        if self.expander is not None:
            try:
                variations = self.expander.expand(clean_q)
            except Exception as e:
                logger.warning(f"Query expansion failed for '{clean_q}': {e}")

        if not variations:
            trans = self._normalize_query_language(clean_q)
            variations = [trans] if trans else [clean_q]

        # Preserve branch positions so configured weights retain their meaning.
        variations = [v.strip() for v in variations if v.strip()][:3]
        return self.encoder.encode_text_batch(variations), variations

    def encode_and_expand_query(self, query: str) -> Tuple[np.ndarray, List[str]]:
        """Compatibility for image blending: use only the faithful primary vector."""
        vecs, variations = self.encode_query_variants(query)
        return vecs[:1], variations

    def retrieve_text(self, query: str, top_k: int = 1000) -> List[Dict[str, Any]]:
        vecs, variations = self.encode_query_variants(query)
        if not variations:
            return []
        rankings = self.retriever.search_batch(vecs, top_k=top_k)
        return weighted_rrf(rankings, config.EXPANSION_WEIGHTS,
                            config.RRF_RANK_CONSTANT)[:top_k]

    def apply_ocr_boost(self, candidates: List[Dict[str, Any]], query: str, ocr_text: str = "") -> List[Dict[str, Any]]:
        """Boost candidate scores if deterministic OCR entities match."""
        if not getattr(config, "ENABLE_OCR_SEARCH", True) or self.text_search is None:
            return candidates

        manual_text = ocr_text.strip()
        if manual_text:
            # Commas separate alternatives; keep phrases intact and ignore empty/duplicate entries.
            ocr_kws = list(dict.fromkeys(
                part.strip().casefold() for part in manual_text.split(",") if part.strip()
            ))
        else:
            ocr_kws = self.text_search.extract_deterministic_keywords(query)
        if not ocr_kws:
            return candidates

        ocr_matches = self.text_search.search_candidates(ocr_kws, boost_score=config.OCR_BOOST_WEIGHT)
        boost_count = 0
        for c in candidates:
            base_score = normalized_score(c)
            c["score"] = base_score / (1.0 + max(0.0, config.OCR_BOOST_WEIGHT))
            c["score_space"] = "unit"
            v_id = c.get("video_id", "")
            k_name = str(c.get("keyframe_name") or c.get("frame_id", ""))
            key = (v_id, k_name)
            if key in ocr_matches:
                match_info = ocr_matches[key]
                c["score"] = (base_score + max(0.0, config.OCR_BOOST_WEIGHT)) / (1.0 + max(0.0, config.OCR_BOOST_WEIGHT))
                c["ocr_match"] = match_info["matched_keyword"]
                boost_count += 1

        if boost_count > 0:
            logger.info(f"[OCR Boost] Boosted {boost_count} candidate frames for keywords: {ocr_kws}")
            candidates.sort(key=lambda x: -float(x.get("score", 0.0)))

        return candidates

    def process_temporal_query(
        self,
        first_query: str,
        second_query: str = "",
        top_k: int = 1000,
        ocr_text: str = ""
    ) -> List[Dict[str, Any]]:
        """Execute single or sequential temporal search with proximity boosting."""
        if not second_query.strip():
            candidates = self.retrieve_text(first_query, top_k=top_k)
            candidates = self.apply_ocr_boost(candidates, first_query, ocr_text)
            return self.format_results(candidates, limit=top_k)

        # Temporal Query: expand and encode both events
        r1 = self.retrieve_text(first_query, top_k=max(top_k * 2, 2000))
        r2 = self.retrieve_text(second_query, top_k=max(top_k * 2, 2000))

        # Group second query candidates by video_id
        r2_by_vid: Dict[str, List[Dict[str, Any]]] = {}
        for item in r2:
            vid = item.get("video_id", "")
            r2_by_vid.setdefault(vid, []).append(item)

        # Score boost for chronological temporal proximity (event 2 follows event 1 in same video)
        MAX_FRAME_GAP = 1500  # ~50 seconds at 30fps
        boosted: List[Dict[str, Any]] = []

        for item in r1:
            item_copy = dict(item)
            vid = item_copy.get("video_id", "")
            frame1 = int(item_copy.get("frame_id", 0))
            base_score = float(item_copy.get("score", 0.0))

            max_boost = 0.0
            if vid in r2_by_vid:
                for target in r2_by_vid[vid]:
                    frame2 = int(target.get("frame_id", 0))
                    gap = frame2 - frame1
                    if 0 < gap <= MAX_FRAME_GAP:
                        target_score = float(target.get("score", 0.0))
                        boost = target_score * (MAX_FRAME_GAP - gap) / MAX_FRAME_GAP
                        if boost > max_boost:
                            max_boost = boost

            item_copy["score"] = (base_score + max_boost) / 2.0
            boosted.append(item_copy)

        boosted.sort(key=lambda x: -float(x.get("score", 0.0)))
        if ocr_text.strip():
            boosted = self.apply_ocr_boost(boosted, "", ocr_text)
        return self.format_results(boosted, limit=top_k)

    def process_multi_event_query(
        self,
        queries: List[str],
        top_k: int = 1000,
        ocr_text: str = ""
    ) -> List[Dict[str, Any]]:
        """Execute single or sequential temporal search across arbitrary N events (TRAKE)."""
        clean_queries = [q.strip() for q in queries if q and q.strip()]
        if not clean_queries:
            return []

        if len(clean_queries) == 1:
            candidates = self.retrieve_text(clean_queries[0], top_k=top_k)
            candidates = self.apply_ocr_boost(candidates, clean_queries[0], ocr_text)
            return self.format_results(candidates, limit=top_k)

        if len(clean_queries) == 2:
            return self.process_temporal_query(clean_queries[0], clean_queries[1], top_k=top_k, ocr_text=ocr_text)

        # For N >= 3 events (TRAKE chained temporal matching)
        event_results = []
        for q in clean_queries:
            res = self.retrieve_text(q, top_k=max(top_k * 2, 2000))
            event_results.append(res)

        # Group subsequent events by video_id
        event_by_vid: List[Dict[str, List[Dict[str, Any]]]] = []
        for res in event_results[1:]:
            by_vid: Dict[str, List[Dict[str, Any]]] = {}
            for item in res:
                vid = item.get("video_id", "")
                by_vid.setdefault(vid, []).append(item)
            event_by_vid.append(by_vid)

        MAX_FRAME_GAP = 1800  # ~60 seconds at 30fps
        boosted: List[Dict[str, Any]] = []

        for item in event_results[0]:
            item_copy = dict(item)
            vid = item_copy.get("video_id", "")
            current_frame = int(item_copy.get("frame_id", 0))
            base_score = float(item_copy.get("score", 0.0))

            total_boost = 0.0
            matched_events = 1

            for by_vid in event_by_vid:
                if vid not in by_vid:
                    break

                best_sub_boost = 0.0
                next_frame = None
                for target in by_vid[vid]:
                    f2 = int(target.get("frame_id", 0))
                    gap = f2 - current_frame
                    if 0 < gap <= MAX_FRAME_GAP:
                        target_score = float(target.get("score", 0.0))
                        boost = target_score * (MAX_FRAME_GAP - gap) / MAX_FRAME_GAP
                        if boost > best_sub_boost:
                            best_sub_boost = boost
                            next_frame = f2

                if next_frame is not None:
                    matched_events += 1
                    total_boost += best_sub_boost
                    current_frame = next_frame
                else:
                    break

            seq_ratio = matched_events / len(clean_queries)
            item_copy["score"] = (base_score + total_boost) * seq_ratio / len(clean_queries)
            boosted.append(item_copy)

        boosted.sort(key=lambda x: -float(x.get("score", 0.0)))
        if ocr_text.strip():
            boosted = self.apply_ocr_boost(boosted, "", ocr_text)
        return self.format_results(boosted, limit=top_k)

    def similarity_search_by_id(self, faiss_idx: int, top_k: int = 1000) -> List[Dict[str, Any]]:
        """Visual similarity search by reconstructing embedding from FAISS index."""
        if self.retriever.index is None:
            raise HTTPException(status_code=500, detail="FAISS index not loaded")

        if faiss_idx < 0 or faiss_idx >= self.retriever.index.ntotal:
            raise HTTPException(status_code=400, detail=f"Index {faiss_idx} out of range (0-{self.retriever.index.ntotal})")

        # Reconstruct normalized vector from FAISS IndexFlatIP
        vec = self.retriever.index.reconstruct(int(faiss_idx)).reshape(1, -1)
        results = self.retriever.search(vec, top_k=top_k)
        return self.format_results(results, limit=top_k)

    def mixed_search(self, image_b64: Optional[str], text: Optional[str], top_k: int = 1000, ocr_text: str = "") -> List[Dict[str, Any]]:
        """Multi-modal search blending image vector and text vector."""
        vecs = []
        if text and text.strip():
            t_vec, _ = self.encode_and_expand_query(text.strip())
            vecs.append(t_vec[0])

        if image_b64:
            pil_img = self.decode_base64_image(image_b64)
            vecs.append(self.encoder.encode_image(pil_img))

        if not vecs:
            return []

        if len(vecs) == 1:
            query_vec = vecs[0].reshape(1, -1)
        else:
            blended = config.MIXED_IMAGE_WEIGHT * vecs[0] + config.MIXED_TEXT_WEIGHT * vecs[1]
            norm = np.linalg.norm(blended)
            query_vec = (blended / (norm + 1e-8)).astype(np.float32).reshape(1, -1)

        results = self.retriever.search(query_vec, top_k=top_k)
        if ocr_text.strip():
            results = self.apply_ocr_boost(results, "", ocr_text)
        return self.format_results(results, limit=top_k)

    def double_image_search(self, img1_b64: str, img2_b64: str, top_k: int = 1000) -> List[Dict[str, Any]]:
        """Search blending two query images."""
        v1 = self.encoder.encode_image(self.decode_base64_image(img1_b64))
        v2 = self.encoder.encode_image(self.decode_base64_image(img2_b64))
        blended = 0.5 * v1 + 0.5 * v2
        norm = np.linalg.norm(blended)
        query_vec = (blended / (norm + 1e-8)).astype(np.float32).reshape(1, -1)

        results = self.retriever.search(query_vec, top_k=top_k)
        return self.format_results(results, limit=top_k)

    def group_search_by_id(self, faiss_idx: int) -> List[Dict[str, Any]]:
        """Retrieve all keyframes of the video belonging to the specified faiss_idx."""
        if self.retriever.meta_df is None:
            return []

        row = self.retriever.meta_df.row(int(faiss_idx), named=True)
        video_id = row.get("video_id", "")
        if not video_id:
            return []

        vid_df = self.retriever.meta_df.filter(pl.col("video_id") == video_id).sort("pts_time")
        candidates = vid_df.to_dicts()
        for cand in candidates:
            cand["score"] = 1.0
        return self.format_results(candidates, limit=len(candidates))


# ==============================================================================
# FastAPI Application Factory
# ==============================================================================

def create_app(llm_model: Optional[str] = None, vlm_model: Optional[str] = None) -> FastAPI:
    app = FastAPI(
        title="AIC-HCMC-26 AI Retrieval Service",
        description="Unified local FastAPI server powered by RTX 4060 GPU and FAISS",
        version="2.0.0"
    )

    # Enable CORS for local web interfaces & dev origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Enable GZip for large JSON result payloads
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Initialize Service EXACTLY ONCE with chosen models
    service = SearchService(llm_model=llm_model, vlm_model=vlm_model)

    # Active websocket tracking
    active_websockets: List[WebSocket] = []

    # --------------------------------------------------------------------------
    # REST Endpoints
    # --------------------------------------------------------------------------

    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "device": service.device,
            "total_vectors": int(service.retriever.index.ntotal) if service.retriever.index else 0,
            "metadata_rows": int(service.retriever.meta_df.height) if service.retriever.meta_df is not None else 0
        }

    @app.get("/config")
    async def get_config():
        return {
            "clip_model": config.CLIP_MODEL_NAME,
            "clip_dim": config.CLIP_DIM,
            "device": service.device,
            "faiss_index_path": str(config.FAISS_INDEX_PATH),
            "metadata_path": str(config.METADATA_PATH),
            "objects_path": str(config.OBJECTS_PATH)
        }

    @app.post("/TextQuery")
    async def text_query_endpoint(request: Request):
        try:
            body = await request.body()
            data = json.loads(body.decode("utf-8"))
            queries = data.get("queries")
            if not queries:
                first_query = data.get("First_query", "")
                next_query = data.get("Next_query", "")
                queries = [q for q in [first_query, next_query] if q and q.strip()]

            if not queries:
                return {
                    "kq": [],
                    "fquery": "",
                    "nquery": "",
                    "queries": [],
                    "total_results": 0
                }

            ocr_text = data.get("ocr_text", "")
            if not isinstance(ocr_text, str):
                raise HTTPException(status_code=422, detail="ocr_text must be a string")
            results = service.process_multi_event_query(queries, ocr_text=ocr_text)

            # Optional VLM Reranking on Top-K if requested
            if (data.get("vlm_rerank", False) or getattr(config, "ENABLE_VLM_RERANK", False)) and service.vlm_reranker is not None:
                if queries:
                    results = service.vlm_reranker.rerank_candidates(
                        results,
                        query=queries[0],
                        top_n=config.VLM_RERANK_TOP_K,
                        vlm_weight=config.VLM_RERANK_WEIGHT,
                        diversity_index=service.retriever.index,
                        diversity_pool=config.VLM_DIVERSITY_POOL,
                        diversity_similarity=config.VLM_DIVERSITY_SIMILARITY,
                        diversity_seconds=config.VLM_DIVERSITY_SECONDS,
                        image_workers=config.VLM_IMAGE_WORKERS,
                    )

            service.last_results = results

            expanded_info = [service.expander.cache.get(q.strip(), []) for q in queries] if service.expander else []
            return {
                "kq": results,
                "fquery": queries[0] if len(queries) > 0 else "",
                "nquery": queries[1] if len(queries) > 1 else "",
                "queries": queries,
                "expanded": expanded_info,
                "total_results": len(results)
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in /TextQuery: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/rerank")
    async def rerank_endpoint(request: Request):
        """Rerank top candidates using Qwen2-VL-7B (4-bit)."""
        try:
            body = await request.body()
            data = json.loads(body.decode("utf-8")) if body else {}
            query = data.get("query", "")
            candidates = data.get("candidates")
            top_n = int(data.get("top_n", config.VLM_RERANK_TOP_K))

            if candidates is None:
                candidates = service.last_results

            if not query and service.expander and service.expander.cache:
                query = list(service.expander.cache.keys())[-1]

            if not candidates or not query:
                return {"kq": candidates or [], "status": "no_candidates_or_query"}

            if service.vlm_reranker is None:
                return {"kq": candidates, "status": "vlm_not_configured"}

            reranked = service.vlm_reranker.rerank_candidates(
                candidates=candidates,
                query=query,
                top_n=top_n,
                vlm_weight=config.VLM_RERANK_WEIGHT,
                diversity_index=service.retriever.index,
                diversity_pool=config.VLM_DIVERSITY_POOL,
                diversity_similarity=config.VLM_DIVERSITY_SIMILARITY,
                diversity_seconds=config.VLM_DIVERSITY_SECONDS,
                image_workers=config.VLM_IMAGE_WORKERS,
            )
            service.last_results = reranked
            return {
                "kq": reranked,
                "status": "success",
                "reranked_count": sum(bool(c.get("vlm_selected")) for c in reranked)
            }
        except Exception as e:
            logger.error(f"Error in /rerank: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/similarity_search")
    async def similarity_search_endpoint(request: Request):
        try:
            body = await request.body()
            data = json.loads(body.decode("utf-8"))
            vector_input = data.get("vector")

            if isinstance(vector_input, int) or (isinstance(vector_input, str) and vector_input.isdigit()):
                faiss_idx = int(vector_input)
                results = service.similarity_search_by_id(faiss_idx)
            else:
                # Direct vector search if an array was provided
                query_vec = np.array(vector_input, dtype=np.float32).reshape(1, -1)
                res = service.retriever.search(query_vec, top_k=1000)
                results = service.format_results(res)

            service.last_results = results
            return {"kq": results, "total_results": len(results)}
        except Exception as e:
            logger.error(f"Error in /similarity_search: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/mixed_query")
    async def mixed_query_endpoint(req: MixedQueryRequest):
        try:
            results = service.mixed_search(req.image, req.text, ocr_text=req.ocr_text)
            service.last_results = results
            return {"kq": results, "total_results": len(results)}
        except Exception as e:
            logger.error(f"Error in /mixed_query: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/double_image_query")
    async def double_image_query_endpoint(req: DoubleImageQueryRequest):
        try:
            if not req.firstImage or not req.secondImage:
                raise HTTPException(status_code=400, detail="Both images required")
            results = service.double_image_search(req.firstImage, req.secondImage)
            service.last_results = results
            return {"kq": results, "total_results": len(results)}
        except Exception as e:
            logger.error(f"Error in /double_image_query: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    # --------------------------------------------------------------------------
    # WebSocket Endpoints matching Baseline Frontend Contracts
    # --------------------------------------------------------------------------

    @app.websocket("/ws")
    async def main_websocket(websocket: WebSocket):
        await websocket.accept()
        active_websockets.append(websocket)
        logger.info("[WebSocket] Main client connected")
        try:
            while True:
                data = await websocket.receive_json()
                msg_type = data.get("type", "")

                if msg_type == "multi_query":
                    queries = data.get("queries", [])
                    text_queries = [q.get("content", "") for q in queries if q.get("type") == "text" and q.get("content")]
                    image_queries = [q.get("content", "") for q in queries if q.get("type") == "image" and q.get("content")]

                    if image_queries and text_queries:
                        results = service.mixed_search(image_queries[0], text_queries[0])
                    elif image_queries and len(image_queries) >= 2:
                        results = service.double_image_search(image_queries[0], image_queries[1])
                    elif image_queries:
                        results = service.mixed_search(image_queries[0], None)
                    elif len(text_queries) >= 2:
                        results = service.process_temporal_query(text_queries[0], text_queries[1])
                    elif text_queries:
                        results = service.process_temporal_query(text_queries[0], "")
                    else:
                        results = []

                    service.last_results = results
                    await websocket.send_json({"kq": results})

                elif msg_type == "text_query":
                    f_q = data.get("firstQuery", "")
                    s_q = data.get("secondQuery", "")
                    results = service.process_temporal_query(f_q, s_q)
                    service.last_results = results
                    await websocket.send_json({"kq": results})

        except WebSocketDisconnect:
            logger.info("[WebSocket] Main client disconnected")
        except Exception as e:
            logger.error(f"[WebSocket] Error: {e}", exc_info=True)
        finally:
            if websocket in active_websockets:
                active_websockets.remove(websocket)

    @app.websocket("/ws/similarity_search")
    async def similarity_websocket(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                data = await websocket.receive_json()
                vector_id = data.get("vector")
                if vector_id is not None:
                    results = service.similarity_search_by_id(int(vector_id))
                    service.last_results = results
                    await websocket.send_json({"kq": results})
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"[WebSocket Similarity] Error: {e}")

    @app.websocket("/ws/filter_query")
    async def filter_websocket(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                data = await websocket.receive_json()
                text_queries = data.get("textQueries", [])
                filters = data.get("filters", [])

                query_text = ""
                for item in text_queries:
                    content = item.get("content", "") if isinstance(item, dict) else str(item)
                    if content:
                        query_text = content
                        break

                required_labels = []
                for filter_group in filters:
                    if isinstance(filter_group, list):
                        for f in filter_group:
                            name = f.get("name", "").strip()
                            if name:
                                required_labels.append(name)

                # Execute search
                candidates = service.kis_task.execute(query_text or "scene", top_k=2000)
                if required_labels and service.object_filter:
                    candidates = service.object_filter.matching_candidates(candidates, required_labels)

                results = service.format_results(candidates, limit=1000)
                service.last_results = results
                await websocket.send_json({"kq": results})
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"[WebSocket Filter] Error: {e}")

    @app.websocket("/ws/group_search")
    async def group_search_websocket(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                data = await websocket.receive_json()
                image_id = data.get("imageId")
                if image_id is not None:
                    results = service.group_search_by_id(int(image_id))
                    await websocket.send_json({"kq": results})
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"[WebSocket GroupSearch] Error: {e}")

    @app.websocket("/ws/pagnition")
    async def pagination_websocket(websocket: WebSocket):
        await websocket.accept()
        PAGE_SIZE = 100
        try:
            while True:
                data = await websocket.receive_json()
                page = int(data.get("page", 0))
                start = page * PAGE_SIZE
                end = start + PAGE_SIZE
                page_results = service.last_results[start:end] if service.last_results else []
                await websocket.send_json({"kq": page_results, "page": page})
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"[WebSocket Pagination] Error: {e}")

    # Auxiliary websockets for UI feature compatibility
    @app.websocket("/ws/share_image")
    @app.websocket("/ws/share_query")
    @app.websocket("/ws/log")
    @app.websocket("/ws/alerts")
    async def auxiliary_websocket(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                data = await websocket.receive_text()
                await websocket.send_text(json.dumps({"status": "ok", "message": "ack"}))
        except WebSocketDisconnect:
            pass

    # --------------------------------------------------------------------------
    # Static UI Files Mounting
    # --------------------------------------------------------------------------
    frontend_dir = REPO_ROOT / "baseline" / "retrievalBaseline" / "frontend"
    if frontend_dir.exists():
        logger.info(f"Mounting baseline frontend at / from {frontend_dir}")
        app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

    return app


# ==============================================================================
# CLI Entrypoint
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AIC 2026 Core Tech FastAPI Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8006, help="Port (default: 8006)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--llm", type=str, default="3b", help="Chọn mô hình SLM: '3b' (Qwen2.5-3B) hoặc '4b' (Qwen3-4B). Mặc định: '3b'")
    parser.add_argument("--vlm", type=str, default="3b", help="Chọn mô hình VLM: '3b' (Qwen2.5-VL-3B), '2b' (Qwen2-VL-2B), '7b'. Mặc định: '3b'")
    args = parser.parse_args()

    chosen_llm = resolve_llm_model(args.llm)
    chosen_vlm = resolve_vlm_model(args.vlm)

    # Initialize app EXACTLY ONCE with the requested models
    app = create_app(llm_model=chosen_llm, vlm_model=chosen_vlm)
    logger.info(f"Starting server with SLM={chosen_llm}, VLM={chosen_vlm} on http://{args.host}:{args.port} ...")
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)
else:
    # Only created when imported as an ASGI module (e.g. `uvicorn src.server:app`)
    app = create_app()
