"""
init_db.py — Initialize Zilliz Cloud (Milvus) specifically for the ReCap pipeline.
For Supabase (PostgreSQL), run the generated 'supabase_schema.sql' in the Supabase SQL Editor.
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

ZILLIZ_URI   = os.getenv("ZILLIZ_URI_RECAP",   getattr(config, "ZILLIZ_URI_RECAP", ""))
ZILLIZ_TOKEN = os.getenv("ZILLIZ_TOKEN_RECAP", getattr(config, "ZILLIZ_TOKEN_RECAP", ""))

CLIP_DIM = 512
MILVUS_COLLECTION = "aic_captions"


def init_zilliz():
    from pymilvus import MilvusClient, DataType
    
    if not ZILLIZ_URI or "your-zilliz" in ZILLIZ_URI:
        print("[Zilliz] Please set ZILLIZ_URI and ZILLIZ_TOKEN in config.py or .env")
        return

    print(f"[Zilliz] Connecting to {ZILLIZ_URI} ...")
    client = MilvusClient(uri=ZILLIZ_URI, token=ZILLIZ_TOKEN)

    if client.has_collection(MILVUS_COLLECTION):
        print(f"[Zilliz] Collection '{MILVUS_COLLECTION}' already exists. Skipping.")
        return

    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("id",            DataType.INT64,        is_primary=True)
    schema.add_field("video_id",      DataType.VARCHAR,      max_length=64)
    schema.add_field("keyframe_name", DataType.VARCHAR,      max_length=64)
    schema.add_field("shot_id",       DataType.INT32)
    schema.add_field("caption",       DataType.VARCHAR,      max_length=2048)
    schema.add_field("vector",        DataType.FLOAT_VECTOR, dim=CLIP_DIM)

    # Zilliz supports AUTOINDEX which automatically optimizes the index for cloud
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="AUTOINDEX",
        metric_type="IP"
    )

    client.create_collection(
        collection_name=MILVUS_COLLECTION,
        schema=schema,
        index_params=index_params,
    )
    print(f"[Zilliz] Collection '{MILVUS_COLLECTION}' created successfully!")


def generate_supabase_sql():
    sql = """-- Run this exactly as is in the Supabase SQL Editor (https://supabase.com/dashboard)
CREATE TABLE aic_captions (
    id SERIAL PRIMARY KEY,
    video_id TEXT NOT NULL,
    keyframe_name TEXT NOT NULL,
    shot_id INTEGER,
    caption TEXT,
    shot_summary TEXT,
    video_summary TEXT,
    transcript TEXT,
    objects TEXT,
    -- This is the Postgres Full-Text Search tsvector column
    fts tsvector GENERATED ALWAYS AS (
        to_tsvector('english', 
            coalesce(caption, '') || ' ' || 
            coalesce(shot_summary, '') || ' ' || 
            coalesce(video_summary, '') || ' ' || 
            coalesce(transcript, '') || ' ' || 
            coalesce(objects, '')
        )
    ) STORED
);

-- Create a fast GIN index for lightning-fast text search
CREATE INDEX aic_captions_fts_idx ON aic_captions USING GIN (fts);

-- Create standard indexes for filtering
CREATE INDEX aic_captions_video_id_idx ON aic_captions (video_id);
"""
    sql_path = Path(__file__).resolve().parent / "supabase_schema.sql"
    with open(sql_path, "w") as f:
        f.write(sql)
    print(f"\n[Supabase] Generated SQL schema at: {sql_path}")
    print("           Please paste the contents of that file into the Supabase SQL Editor.")


if __name__ == "__main__":
    init_zilliz()
    generate_supabase_sql()
