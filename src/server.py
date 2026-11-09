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
from typing import Any, Dict, List, Optional

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
from src.online_pipeline.object_filter import ObjectFilter
from src.online_pipeline.query_encoder import QueryEncoder
from src.online_pipeline.retrieval import RetrievalEngine
from src.tasks.kis_t import KIStask
from src.tasks.trake import TrakeTask

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


class SimilaritySearchRequest(BaseModel):
    vector: Any  # Accepts int faiss_idx or list of floats


class MixedQueryRequest(BaseModel):
    image: Optional[str] = None  # Base64 encoded image
    text: Optional[str] = ""


class DoubleImageQueryRequest(BaseModel):
    firstImage: Optional[str] = None
    secondImage: Optional[str] = None


class VQAQueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 50


# ==============================================================================
# Service Container
# ==============================================================================

class SearchService:
    def __init__(self):
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

        logger.info(f"Loading Object Filter ({config.OBJECTS_PATH})...")
        self.object_filter = ObjectFilter(object_path=config.OBJECTS_PATH)

        logger.info("Initializing KIStask and TrakeTask...")
        self.kis_task = KIStask(
            encoder=self.encoder,
            retriever=self.retriever,
            object_filter=self.object_filter
        )
        self.trake_task = TrakeTask(
            encoder=self.encoder,
            retriever=self.retriever
        )

        # Cache of latest results for pagination
        self.last_results: List[Dict[str, Any]] = []

        logger.info("Service initialized successfully.")

    def format_results(self, candidates: List[Dict[str, Any]], limit: int = 1000) -> List[Dict[str, Any]]:
        """Format candidate dictionary to Milvus-compatible JSON structure expected by frontend."""
        formatted = []
        for idx, cand in enumerate(candidates[:limit]):
            v_id = str(cand.get("video_id", ""))
            k_name = str(cand.get("keyframe_name") or cand.get("frame_id", ""))
            pts_time = float(cand.get("pts_time", 0.0))
            score = float(cand.get("score", 0.0))
            f_idx = int(cand.get("faiss_idx", idx))

            formatted.append({
                "id": f_idx,
                "distance": score,
                "score": score,
                "entity": {
                    "video": v_id,
                    "frame_id": k_name,
                    "time": pts_time,
                    "path": f"keyframes/{v_id}/{k_name}.jpg"
                }
            })
        return formatted

    def decode_base64_image(self, data_url_or_b64: str) -> Image.Image:
        """Decode base64 image data URL to PIL Image."""
        if "," in data_url_or_b64:
            data_url_or_b64 = data_url_or_b64.split(",", 1)[1]
        img_bytes = base64.b64decode(data_url_or_b64)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")

    def process_temporal_query(
        self,
        first_query: str,
        second_query: str = "",
        top_k: int = 1000
    ) -> List[Dict[str, Any]]:
        """Execute single or sequential temporal search with proximity boosting."""
        if not second_query.strip():
            # Standard single query via KIStask
            candidates = self.kis_task.execute(first_query, top_k=top_k)
            return self.format_results(candidates, limit=top_k)

        # Temporal Query: retrieve candidates for both events
        v1 = self.encoder.encode_text(first_query)
        v2 = self.encoder.encode_text(second_query)

        r1 = self.retriever.search(v1, top_k=max(top_k * 2, 2000))
        r2 = self.retriever.search(v2, top_k=max(top_k * 2, 2000))

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

            item_copy["score"] = base_score + max_boost
            boosted.append(item_copy)

        boosted.sort(key=lambda x: -float(x.get("score", 0.0)))
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

    def mixed_search(self, image_b64: Optional[str], text: Optional[str], top_k: int = 1000) -> List[Dict[str, Any]]:
        """Multi-modal search blending image vector and text vector."""
        vecs = []
        if text and text.strip():
            vecs.append(self.encoder.encode_text(text.strip()))

        if image_b64:
            pil_img = self.decode_base64_image(image_b64)
            vecs.append(self.encoder.encode_image(pil_img))

        if not vecs:
            return []

        if len(vecs) == 1:
            query_vec = vecs[0].reshape(1, -1)
        else:
            blended = 0.5 * vecs[0] + 0.5 * vecs[1]
            norm = np.linalg.norm(blended)
            query_vec = (blended / (norm + 1e-8)).astype(np.float32).reshape(1, -1)

        results = self.retriever.search(query_vec, top_k=top_k)
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

def create_app() -> FastAPI:
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

    # Initialize Service
    service = SearchService()

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
            first_query = data.get("First_query", "")
            next_query = data.get("Next_query", "")

            results = service.process_temporal_query(first_query, next_query)
            service.last_results = results

            return {
                "kq": results,
                "fquery": first_query,
                "nquery": next_query,
                "total_results": len(results)
            }
        except Exception as e:
            logger.error(f"Error in /TextQuery: {e}", exc_info=True)
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
            results = service.mixed_search(req.image, req.text)
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
                if required_labels:
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
    args = parser.parse_args()

    app = create_app()
    logger.info(f"Starting server on http://{args.host}:{args.port} ...")
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)
