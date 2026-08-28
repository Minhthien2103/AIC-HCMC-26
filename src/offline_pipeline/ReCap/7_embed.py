"""
Stage 7 - Embed captions with MobileCLIP text encoder and ingest into Milvus + ElasticSearch.
Input:  data/captions.parquet
        data/shot_summaries.parquet
        data/video_summaries.parquet
Output: Milvus collection 'aic_captions' (vectors)
        ElasticSearch index 'aic_captions' (text)
"""
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

MILVUS_URI = os.getenv("MILVUS_URI", config.MILVUS_URI)
ES_URL     = os.getenv("ES_URL",     config.ES_URL)

CAPTIONS_PATH       = config.DATA_DIR / "captions.parquet"
SHOT_SUMMARIES_PATH = config.DATA_DIR / "shot_summaries.parquet"
VIDEO_SUMMARIES_PATH = config.DATA_DIR / "video_summaries.parquet"

MILVUS_COLLECTION = "aic_captions"
ES_INDEX          = "aic_captions"
BATCH_SIZE        = 256


def _load_clip_text_encoder():
    """Load MobileCLIP or CLIP text encoder for embedding captions."""
    try:
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms(
            config.CLIP_MODEL_NAME, pretrained=config.CLIP_PRETRAINED
        )
        model.eval()
        tokenizer = open_clip.get_tokenizer(config.CLIP_MODEL_NAME)
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        def encode(texts: list[str]) -> np.ndarray:
            tokens = tokenizer(texts).to(device)
            with torch.no_grad():
                feats = model.encode_text(tokens)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats.cpu().numpy().astype("float32")

        print(f"[Stage7] CLIP encoder loaded on {device}.")
        return encode
    except Exception as e:
        print(f"[Stage7] Encoder load failed: {e}")
        raise


def _ingest_milvus(df: pd.DataFrame, encode_fn):
    from pymilvus import MilvusClient
    client = MilvusClient(uri=MILVUS_URI)

    if not client.has_collection(MILVUS_COLLECTION):
        print(f"[Stage7] Collection not found. Run init_db.py first.")
        return

    total = 0
    for i in range(0, len(df), BATCH_SIZE):
        batch = df.iloc[i : i + BATCH_SIZE]
        texts = batch["caption"].fillna("").tolist()
        vectors = encode_fn(texts)  # shape (B, 512)

        data = [
            {
                "video_id":      row.video_id,
                "keyframe_name": str(row.keyframe_name),
                "shot_id":       int(getattr(row, "shot_id", 0)),
                "caption":       str(row.caption) if row.caption else "",
                "vector":        vectors[j].tolist(),
            }
            for j, row in enumerate(batch.itertuples())
        ]
        client.insert(collection_name=MILVUS_COLLECTION, data=data)
        total += len(data)
        print(f"  [Milvus] Inserted {total}/{len(df)}")

    print(f"[Stage7] Milvus ingestion done. {total} vectors inserted.")


def _ingest_elasticsearch(df_cap: pd.DataFrame, df_shots: pd.DataFrame, df_vids: pd.DataFrame):
    from elasticsearch import Elasticsearch
    from elasticsearch.helpers import bulk

    es = Elasticsearch(ES_URL)

    # Build lookup tables
    shot_summary_map = {}
    if df_shots is not None:
        for row in df_shots.itertuples():
            shot_summary_map[(row.video_id, int(row.shot_id))] = str(row.summary)

    vid_summary_map = {}
    if df_vids is not None:
        for row in df_vids.itertuples():
            vid_summary_map[row.video_id] = str(row.summary)

    def _actions():
        for row in df_cap.itertuples():
            vid = row.video_id
            shot_id = int(getattr(row, "shot_id", 0))
            yield {
                "_index": ES_INDEX,
                "_id":    f"{vid}_{row.keyframe_name}",
                "_source": {
                    "video_id":      vid,
                    "keyframe_name": str(row.keyframe_name),
                    "shot_id":       shot_id,
                    "caption":       str(row.caption) if row.caption else "",
                    "shot_summary":  shot_summary_map.get((vid, shot_id), ""),
                    "video_summary": vid_summary_map.get(vid, ""),
                    "transcript":    "",  # Filled when transcripts.parquet exists
                    "objects":       "",  # Filled by YOLO pipeline
                }
            }

    success, failed = bulk(es, _actions(), raise_on_error=False)
    print(f"[Stage7] ES ingestion done. {success} ok, {failed} failed.")


def run_stage7():
    if not CAPTIONS_PATH.exists():
        print("[Stage7] captions.parquet not found. Run Stage 4 first.")
        return

    df_cap   = pd.read_parquet(CAPTIONS_PATH)
    df_shots = pd.read_parquet(SHOT_SUMMARIES_PATH) if SHOT_SUMMARIES_PATH.exists() else None
    df_vids  = pd.read_parquet(VIDEO_SUMMARIES_PATH) if VIDEO_SUMMARIES_PATH.exists() else None

    print(f"[Stage7] Ingesting {len(df_cap)} keyframes into Milvus and ElasticSearch...")

    encode_fn = _load_clip_text_encoder()
    _ingest_milvus(df_cap, encode_fn)
    _ingest_elasticsearch(df_cap, df_shots, df_vids)

    print("[Stage7] Done.")


if __name__ == "__main__":
    run_stage7()