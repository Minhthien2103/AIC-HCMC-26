"""
FastAPI Vector Search Service
===========================

A high-performance vector search service using CLIP models for image-text similarity search.
Supports temporal queries and provides both REST API and WebSocket interfaces.

Dependencies:
- FastAPI, torch, open_clip
- Milvus vector database
- PIL for image processing
"""

import os
import json
import time
import logging
import asyncio
import base64
from io import BytesIO
from typing import List, Optional, Dict, Any
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

# FastAPI imports
from fastapi import FastAPI, File, UploadFile, Request, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ML/AI imports
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import open_clip

# Vector database imports
from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema, Collection

# Configuration Management
# ========================

@dataclass
class ModelConfig:
    """Configuration for ML models"""
    clip_model_name: str = "ViT-H-14-378-quickgelu"
    clip_pretrained: str = "dfn5b"
    device: str = "cuda"  # auto-detected if cuda available

@dataclass
class DatabaseConfig:
    """Configuration for Milvus vector database"""
    host: str = "192.168.20.156"
    port: int = 19530
    database: str = "default"
    collection_name: str = "AIC_2024_1"
    search_limit: int = 3000
    replica_number: int = 1

@dataclass
class ServerConfig:
    """Configuration for FastAPI server"""
    cors_origins: str = "http://localhost:8007,https://localhost:8005,https://localhost:8443"
    max_workers: int = 4
    log_level: str = "INFO"
    gzip_minimum_size: int = 1000

class Config:
    """Main configuration class that loads from environment variables or config file"""

    def __init__(self, config_file: str = None):
        # Load from config file if provided
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config_data = json.load(f)
        else:
            config_data = {}

        # Initialize configurations with env variables or config file values
        self.model = ModelConfig(
            clip_model_name=os.getenv("CLIP_MODEL_NAME", config_data.get("clip_model_name", "ViT-H-14-378-quickgelu")),
            clip_pretrained=os.getenv("CLIP_PRETRAINED", config_data.get("clip_pretrained", "dfn5b")),
            device=os.getenv("DEVICE", config_data.get("device", "cuda"))
        )

        self.database = DatabaseConfig(
            host=os.getenv("MILVUS_HOST", config_data.get("milvus_host", "192.168.20.156")),
            port=int(os.getenv("MILVUS_PORT", config_data.get("milvus_port", 19530))),
            database=os.getenv("MILVUS_DATABASE", config_data.get("milvus_database", "default")),
            collection_name=os.getenv("COLLECTION_NAME", config_data.get("collection_name", "AIC_2024_1")),
            search_limit=int(os.getenv("SEARCH_LIMIT", config_data.get("search_limit", 3000))),
            replica_number=int(os.getenv("REPLICA_NUMBER", config_data.get("replica_number", 1)))
        )

        self.server = ServerConfig(
            cors_origins=os.getenv("CORS_ORIGINS", config_data.get("cors_origins", "http://localhost:8007,https://localhost:8005,https://localhost:8443")),
            max_workers=int(os.getenv("MAX_WORKERS", config_data.get("max_workers", 4))),
            log_level=os.getenv("LOG_LEVEL", config_data.get("log_level", "INFO")),
            gzip_minimum_size=int(os.getenv("GZIP_MIN_SIZE", config_data.get("gzip_minimum_size", 1000)))
        )

        # Auto-detect device if cuda specified but not available
        if self.model.device == "cuda" and not torch.cuda.is_available():
            self.model.device = "cpu"
            logging.warning("CUDA requested but not available, falling back to CPU")

# Pydantic Models for API
# ======================

class TextQuery(BaseModel):
    """Model for text query requests"""
    First_query: str
    Next_query: str = ""

class Filter(BaseModel):
    """Model for search filters"""
    name: str
    number: str

class FilterQuery(BaseModel):
    """Model for filtered query requests"""
    filterText: Optional[str] = None
    filters: List[Filter]
    query: str

# Global Application State
# =======================

class VectorSearchService:
    """Main service class that encapsulates all functionality"""

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(self.config.model.device)

        # Initialize logging
        logging.basicConfig(level=getattr(logging, self.config.server.log_level))
        self.logger = logging.getLogger(__name__)

        # Initialize thread pool for CPU-bound tasks
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.server.max_workers)

        # Initialize models and database connection
        self._initialize_models()
        self._initialize_database()

        # WebSocket connections
        self.active_connections: List[WebSocket] = []

        # Common query cache for performance
        self.common_queries = ["person", "car", "building"]
        self.precomputed_tokens = {}

    def _initialize_models(self):
        """Initialize ML models (CLIP)"""
        self.logger.info("Initializing ML models...")

        # Initialize OpenAI CLIP model
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            self.config.model.clip_model_name,
            pretrained=self.config.model.clip_pretrained
        )
        self.clip_model = self.clip_model.to(self.device)
        self.clip_tokenizer = open_clip.get_tokenizer(self.config.model.clip_model_name)

        # Precompute common query tokens for performance
        self.precomputed_tokens = {
            query: self.clip_tokenizer([query]).to(self.device)
            for query in self.common_queries
        }

        self.logger.info("Models initialized successfully")

    def _initialize_database(self):
        """Initialize Milvus database connection"""
        self.logger.info("Initializing database connection...")

        # Create Milvus client
        self.milvus_client = MilvusClient(
            uri=f"http://{self.config.database.host}:{self.config.database.port}",
            db=self.config.database.database
        )

        # Load collection
        try:
            self.milvus_client.load_collection(
                collection_name=self.config.database.collection_name,
                replica_number=self.config.database.replica_number
            )

            # Check load state
            load_state = self.milvus_client.get_load_state(
                collection_name=self.config.database.collection_name
            )
            self.logger.info(f"Collection {self.config.database.collection_name} load state: {load_state}")

        except Exception as e:
            self.logger.error(f"Failed to load collection: {e}")
            raise

        self.logger.info("Database connection initialized successfully")

    @lru_cache(maxsize=1000)
    def encode_clip_text(self, query: str) -> torch.Tensor:
        """
        Encode text using CLIP model with caching

        Args:
            query: Text query to encode

        Returns:
            Normalized text features tensor
        """
        text_inputs = self.precomputed_tokens.get(query) or self.clip_tokenizer([query]).to(self.device)

        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_inputs)
            return F.normalize(text_features, p=2, dim=-1)

    @staticmethod
    def log_execution_time(func):
        """Decorator to log function execution time"""
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            result = await func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            logging.getLogger(__name__).info(f"{func.__name__} executed in {execution_time:.4f} seconds")
            return result
        return wrapper

    @log_execution_time
    async def query_milvus(self, query_vector: torch.Tensor, milvus_filter=None, limit: int = None) -> List[Any]:
        """
        Query Milvus vector database

        Args:
            query_vector: Query vector tensor
            milvus_filter: Optional filter for search
            limit: Maximum number of results

        Returns:
            List of search results
        """
        if limit is None:
            limit = self.config.database.search_limit

        return await asyncio.to_thread(
            self.milvus_client.search,
            collection_name=self.config.database.collection_name,
            anns_field="embedding",  # Assuming the vector field is named 'embedding'
            data=[query_vector.tolist()[0]],
            limit=limit,
            output_fields=['path', 'time', 'frame_id'],
            search_params={
                "metric_type": "IP",
                "params": {
                    "itopk_size": 4096,
                    "search_width": 2,
                    "min_iterations": 0,
                    "max_iterations": 0,
                    "team_size": 0
                }
            }
        )

    async def process_temporal_query(self, first_query: str, second_query: str = "") -> List[Any]:
        """
        Process temporal query with two sequential text queries

        Args:
            first_query: First temporal query
            second_query: Optional second temporal query

        Returns:
            List of ranked search results
        """
        start_time = time.time()

        try:
            # Encode queries in parallel if both exist
            if second_query:
                first_encoded, second_encoded = await asyncio.gather(
                    asyncio.to_thread(self.encode_clip_text, first_query),
                    asyncio.to_thread(self.encode_clip_text, second_query)
                )

                # Query database in parallel
                fkq, nkq = await asyncio.gather(
                    self.query_milvus(first_encoded),
                    self.query_milvus(second_encoded)
                )

                # Process temporal relationships using vectorized operations
                result = self._process_temporal_relationships(fkq, nkq)
            else:
                # Single query processing
                first_encoded = await asyncio.to_thread(self.encode_clip_text, first_query)
                fkq = await self.query_milvus(first_encoded)
                result = fkq[:1000]

            return result

        except Exception as e:
            self.logger.error(f"Error in temporal query processing: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            end_time = time.time()
            self.logger.info(f"Temporal query executed in {end_time - start_time:.4f} seconds")

    def _process_temporal_relationships(self, first_results: List[Any], second_results: List[Any]) -> List[Any]:
        """
        Process temporal relationships between two query results using vectorized operations

        Args:
            first_results: Results from first query
            second_results: Results from second query

        Returns:
            Reranked results based on temporal relationships
        """
        # Extract data and convert to PyTorch tensors for faster computation
        fkq_data = torch.tensor([
            [int(item.entity['frame_id']), item.score, hash(item.entity['video'])]
            for item in first_results
        ], device=self.device)

        nkq_data = torch.tensor([
            [int(item.entity['frame_id']), item.score, hash(item.entity['video'])]
            for item in second_results
        ], device=self.device)

        # Vectorized temporal relationship computation
        frame_diff = nkq_data[:, None, 0] - fkq_data[None, :, 0]
        same_video_mask = fkq_data[None, :, 2] == nkq_data[:, None, 2]
        valid_frame_diff_mask = (frame_diff > 0) & (frame_diff <= 1500) & same_video_mask

        # Calculate score boost based on temporal proximity
        score_increase = nkq_data[:, None, 1] * (1500 - frame_diff) / 1500
        score_increase = torch.where(valid_frame_diff_mask, score_increase, torch.zeros_like(score_increase))

        # Update scores with temporal boost
        fkq_data[:, 1] += score_increase.max(dim=0).values
        scores = fkq_data[:, 1].cpu().numpy()

        # Sort and return top results
        sorted_indices = np.argsort(scores)[::-1][:1000]
        return [first_results[i] for i in sorted_indices]

# FastAPI Application Setup
# ========================

def create_app(config_file: str = None) -> FastAPI:
    """
    Create and configure FastAPI application

    Args:
        config_file: Optional path to configuration file

    Returns:
        Configured FastAPI application
    """
    # Load configuration
    config = Config(config_file)

    # Initialize service
    service = VectorSearchService(config)

    # Create FastAPI app
    app = FastAPI(
        title="Vector Search Service",
        description="High-performance vector search service with CLIP models",
        version="1.0.0"
    )

    # Configure CORS
    origins = config.server.cors_origins.split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
        max_age=3600,
    )

    # Add GZip compression
    app.add_middleware(
        GZipMiddleware,
        minimum_size=config.server.gzip_minimum_size
    )

    # API Endpoints
    # =============

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time queries"""
        await websocket.accept()
        service.active_connections.append(websocket)
        service.logger.info("WebSocket connection accepted")

        try:
            while True:
                data = await websocket.receive_json()
                if data['type'] == 'text_query':
                    result = await service.process_temporal_query(
                        data['firstQuery'],
                        data.get('secondQuery', '')
                    )
                    await websocket.send_json({"kq": result})
        except WebSocketDisconnect:
            service.logger.info("WebSocket disconnected")
            service.active_connections.remove(websocket)
        except Exception as e:
            service.logger.error(f"Error in WebSocket: {str(e)}")
            if websocket in service.active_connections:
                service.active_connections.remove(websocket)

    @app.post("/TextQuery")
    async def text_query_endpoint(request: Request):
        """
        REST endpoint for text queries

        Accepts JSON payload with First_query and optional Next_query
        Returns ranked search results
        """
        try:
            body = await request.body()
            temporal_search = json.loads(body)

            first_query = temporal_search['First_query']
            next_query = temporal_search.get('Next_query', '')

            # Process the query
            result = await service.process_temporal_query(first_query, next_query)

            return {
                "kq": result[0] if result else None,
                "fquery": first_query,
                "nquery": next_query,
                "total_results": len(result)
            }

        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON payload")
        except KeyError as e:
            raise HTTPException(status_code=400, detail=f"Missing required field: {e}")
        except Exception as e:
            service.logger.error(f"Error in text query: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")

    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "models_loaded": True,
            "database_connected": True,
            "active_connections": len(service.active_connections)
        }

    @app.get("/config")
    async def get_config():
        """Get current configuration (excluding sensitive data)"""
        return {
            "model_config": {
                "clip_model": config.model.clip_model_name,
                "device": config.model.device
            },
            "database_config": {
                "collection_name": config.database.collection_name,
                "search_limit": config.database.search_limit
            },
            "server_config": {
                "max_workers": config.server.max_workers,
                "log_level": config.server.log_level
            }
        }

    return app

# Application Entry Point
# =======================

if __name__ == "__main__":
    import uvicorn

    # Load configuration file path from environment or use default
    config_file = os.getenv("CONFIG_FILE", "config.json")

    # Create the application
    app = create_app(config_file)

    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False  # Set to True for development
    )