"""
init_db.py — Initialize Milvus and ElasticSearch schemas.
Run once before ingesting any data.

Usage:
    python src/database/init_db.py
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# ── Config ────────────────────────────────────────────────────────────
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
ES_URL     = os.getenv("ES_URL",     "http://localhost:9200")

# Mobile CLIP text encoder output dimension
CLIP_DIM = 512

MILVUS_COLLECTION = "aic_captions"
ES_INDEX          = "aic_captions"

# ── Milvus ────────────────────────────────────────────────────────────
def init_milvus():
    from pymilvus import MilvusClient, DataType
    print(f"[Milvus] Connecting to {MILVUS_URI} ...")
    client = MilvusClient(uri=MILVUS_URI)

    if client.has_collection(MILVUS_COLLECTION):
        print(f"[Milvus] Collection '{MILVUS_COLLECTION}' already exists. Skipping.")
        return

    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("id",            DataType.INT64,        is_primary=True)
    schema.add_field("video_id",      DataType.VARCHAR,      max_length=64)
    schema.add_field("keyframe_name", DataType.VARCHAR,      max_length=64)
    schema.add_field("shot_id",       DataType.INT32)
    schema.add_field("caption",       DataType.VARCHAR,      max_length=2048)
    schema.add_field("vector",        DataType.FLOAT_VECTOR, dim=CLIP_DIM)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="IVF_FLAT",
        metric_type="IP",
        params={"nlist": 1024},
    )

    client.create_collection(
        collection_name=MILVUS_COLLECTION,
        schema=schema,
        index_params=index_params,
    )
    print(f"[Milvus] Collection '{MILVUS_COLLECTION}' created (dim={CLIP_DIM}).")


# ── ElasticSearch ─────────────────────────────────────────────────────
def init_elasticsearch():
    from elasticsearch import Elasticsearch
    print(f"[ES] Connecting to {ES_URL} ...")
    es = Elasticsearch(ES_URL)

    if es.indices.exists(index=ES_INDEX):
        print(f"[ES] Index '{ES_INDEX}' already exists. Skipping.")
        return

    mapping = {
        "mappings": {
            "properties": {
                "video_id":      {"type": "keyword"},
                "keyframe_name": {"type": "keyword"},
                "shot_id":       {"type": "integer"},
                "caption":       {"type": "text", "analyzer": "english"},
                "transcript":    {"type": "text", "analyzer": "english"},
                "objects":       {"type": "text", "analyzer": "english"},
                "shot_summary":  {"type": "text", "analyzer": "english"},
                "video_summary": {"type": "text", "analyzer": "english"},
            }
        },
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
        }
    }

    es.indices.create(index=ES_INDEX, body=mapping)
    print(f"[ES] Index '{ES_INDEX}' created.")


# ── Main ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_milvus()
    init_elasticsearch()
    print("\nAll databases initialized successfully!")