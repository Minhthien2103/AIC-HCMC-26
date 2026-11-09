#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple

import torch
from tqdm import tqdm
from PIL import Image
import torch.multiprocessing as mp

from pymilvus import (
    connections, utility, Collection, CollectionSchema,
    FieldSchema, DataType
)

import open_clip


# -------------------------
# Worker
# -------------------------
def encode_worker(
    rank: int,
    world_size: int,
    indexed_paths: List[Tuple[int, Path]],
    device_ids: List[int],
    milvus_host: str,
    milvus_port: str,
    collection_name: str,
    batch_size: int,
    flush_interval: int,
    model_name: str,
    pretrained: str,
    webp_field_names: Tuple[str, str, str, str, str],  # ("id","filepath","embedding","video_id","frame_id")
):
    id_f, path_f, emb_f, vid_f, frame_f = webp_field_names

    # --- device init
    if torch.cuda.is_available() and len(device_ids) > 0:
        device = torch.device(f"cuda:{device_ids[rank]}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # --- model
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device).eval()

    # --- milvus connection
    connections.connect(host=milvus_host, port=milvus_port)
    collection = Collection(name=collection_name)  # do NOT load for writing

    # --- sharding the work
    my_paths = indexed_paths[rank::world_size]

    buffer = {id_f: [], path_f: [], emb_f: [], vid_f: [], frame_f: [], "images": []}
    since_flush = 0

    for _id, path in tqdm(my_paths, desc=f"[Worker {rank}]"):
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"[Worker {rank}] Failed to open {path}: {e}")
            try:
                with open("fail_load.txt", "a+", encoding="utf-8") as f:
                    f.write(str(path) + "\n")
            except Exception:
                pass
            continue

        buffer["images"].append(preprocess(img))
        buffer[id_f].append(_id)
        buffer[path_f].append(str(path))
        buffer[vid_f].append(path.parent.parent.name)  # <video>/keyframes/file.webp
        try:
            fid = int(path.stem.replace("keyframe_", ""))
        except Exception:
            # fallback if custom naming
            try:
                fid = int(Path(path).stem.split("_")[-1])
            except Exception:
                fid = _id
        buffer[frame_f].append(fid)

        if len(buffer["images"]) >= batch_size:
            _flush_batch(collection, buffer, device, id_f, path_f, emb_f, vid_f, frame_f)
            since_flush += batch_size
            if since_flush >= flush_interval:
                collection.flush()
                since_flush = 0

    # tail
    if buffer["images"]:
        _flush_batch(collection, buffer, device, id_f, path_f, emb_f, vid_f, frame_f)
        collection.flush()

    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"[Worker {rank}] Done.")


@torch.no_grad()
def _flush_batch(collection, buffer, device, id_f, path_f, emb_f, vid_f, frame_f):
    try:
        inputs = torch.stack(buffer["images"]).to(device, non_blocking=True)
        # NOTE: we do NOT normalize here; Milvus COSINE handles similarity
        model = collection._get_connection()  # not used; keep local ref quiet
    except Exception:
        pass  # noqa

    # We cached the model via closure (encode_worker scope), so we need to fetch it:
    # Better: pass model via arguments – but TorchScript/MP pickling is heavy.
    # Workaround: store encode in a global. It's simple and safe.
    global _ENCODE_CACHE
    if "_ENCODE_CACHE" not in globals():
        _ENCODE_CACHE = {}
    if "encode_image" not in _ENCODE_CACHE:
        raise RuntimeError("Internal encode function missing. Please do not modify _flush_batch separately.")

    embs = _ENCODE_CACHE["encode_image"](inputs).cpu().tolist()

    try:
        collection.insert(
            [
                buffer[id_f],
                buffer[path_f],
                embs,
                buffer[vid_f],
                buffer[frame_f],
            ]
        )
    except Exception as e:
        print(f"[Insert] Error: {e}")
    finally:
        buffer.clear()
        buffer.update({id_f: [], path_f: [], emb_f: [], vid_f: [], frame_f: [], "images": []})


# -------------------------
# Helper: prepare encode fn in global scope (workaround for MP pickling)
# -------------------------
def _prepare_encode_cache(model_name: str, pretrained: str, device: torch.device):
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device).eval()

    @torch.no_grad()
    def _encode(x: torch.Tensor) -> torch.Tensor:
        return model.encode_image(x)

    globals()["_ENCODE_CACHE"] = {"encode_image": _encode, "model": model}


# -------------------------
# Schema utils
# -------------------------
def ensure_collection(
    collection_name: str,
    dimension: int,
    milvus_host: str,
    milvus_port: str,
    recreate: bool,
    index_params: dict,
    field_names=("id", "filepath", "embedding", "video_id", "frame_id"),
):
    id_f, path_f, emb_f, vid_f, frame_f = field_names

    connections.connect(host=milvus_host, port=milvus_port)
    if recreate and utility.has_collection(collection_name):
        print(f"⚠️ Dropping existing collection: {collection_name}")
        utility.drop_collection(collection_name)

    if not utility.has_collection(collection_name):
        print(f"Creating collection: {collection_name}")
        schema = CollectionSchema(
            fields=[
                FieldSchema(name=id_f, dtype=DataType.INT64, is_primary=True, auto_id=False),
                FieldSchema(name=path_f, dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name=emb_f, dtype=DataType.FLOAT_VECTOR, dim=dimension),
                FieldSchema(name=vid_f, dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name=frame_f, dtype=DataType.INT64),
            ],
            description="Keyframe embeddings (OpenCLIP) for video retrieval",
        )
        Collection(name=collection_name, schema=schema)
    else:
        print(f"Collection exists: {collection_name}")

    # Don’t build index yet; do it after bulk insert (faster).
    # Also don’t load() for inserting.


# -------------------------
# Discovery
# -------------------------
def discover_images(root: Path, glob_pat: str, start_index: int) -> List[Path]:
    # Default searches **/keyframes/*.webp to match your extractor
    paths = sorted(root.glob(glob_pat))
    if start_index > 0:
        paths = paths[start_index:]
    return paths


# -------------------------
# Build index (post-insert)
# -------------------------
def build_index_and_load(collection_name: str, index_params: dict, milvus_host: str, milvus_port: str, field_name="embedding"):
    connections.connect(host=milvus_host, port=milvus_port)
    collection = Collection(name=collection_name)
    print("Flushing before indexing...")
    collection.flush()

    has_index = any(idx.field_name == field_name for idx in collection.indexes)
    if not has_index:
        print("Creating index...")
        collection.create_index(field_name, index_params)
    else:
        print("Index already exists; skipping create.")

    print("Loading collection into memory for queries...")
    collection.load()
    print("Index ready.")


# -------------------------
# Argparse + Main
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Milvus indexer for keyframe WEBP files encoded by OpenCLIP.")
    # IO & Data
    p.add_argument("--root", type=str, required=True,
                   help="Root folder containing per-video subfolders with keyframes/.")
    p.add_argument("--glob", type=str, default="**/keyframes/*.webp",
                   help="Glob pattern relative to root (default: **/keyframes/*.webp).")
    p.add_argument("--start-index", type=int, default=0, help="Skip the first N files (after sort).")
    p.add_argument("--id-offset", type=int, default=0, help="Add this offset to assigned integer IDs.")

    # Milvus
    p.add_argument("--collection-name", type=str, default="AIC25_fullbatch1")
    p.add_argument("--host", type=str, default="192.168.20.156")
    p.add_argument("--port", type=str, default="19530")
    p.add_argument("--recreate", action="store_true", help="Drop and recreate the collection.")
    p.add_argument("--build-index", action="store_true", help="Build HNSW index after inserts and load collection.")

    # Embeddings / Model
    p.add_argument("--model", type=str, default="ViT-H-14-378-quickgelu")
    p.add_argument("--pretrained", type=str, default="dfn5b")
    p.add_argument("--dimension", type=int, default=-1,
                   help="Embedding dimension. Use -1 to auto-detect from the model (recommended).")

    # Performance
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--flush-interval", type=int, default=2000)
    p.add_argument("--workers", type=str, default="auto",
                   help="'auto' = #GPUs; integer = explicit #workers; 'cpu' = 1 worker on CPU.")
    return p.parse_args()


def auto_detect_dim(model_name: str, pretrained: str, device: torch.device) -> int:
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device).eval()
    # Most OpenCLIP models expose image output dim via model.visual.output_dim or model.embed_dim
    dim = getattr(model, "embed_dim", None)
    if dim is None and hasattr(model, "visual") and hasattr(model.visual, "output_dim"):
        dim = model.visual.output_dim
    if dim is None:
        # fallback: run a dummy forward
        import torch
        dummy = torch.zeros(1, 3, 224, 224, device=device)
        with torch.no_grad():
            out = model.encode_image(dummy)
        dim = out.shape[-1]
    return int(dim)


def main():
    args = parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        print(f"Root not found: {root}")
        sys.exit(1)

    # Workers & devices
    cuda_count = torch.cuda.device_count()
    if args.workers == "cpu":
        device_ids = []
        world_size = 1
        use_cpu = True
    elif args.workers == "auto":
        if cuda_count > 0:
            device_ids = list(range(cuda_count))
            world_size = len(device_ids)
            use_cpu = False
        else:
            device_ids = []
            world_size = 1
            use_cpu = True
    else:
        # explicit int
        try:
            n = int(args.workers)
            if n <= 0:
                raise ValueError
        except ValueError:
            print("--workers must be 'auto', 'cpu', or a positive integer.")
            sys.exit(1)
        if cuda_count == 0:
            device_ids = []
            world_size = 1
            use_cpu = True
        else:
            device_ids = list(range(min(n, cuda_count)))
            world_size = len(device_ids)
            use_cpu = False

    device = torch.device("cpu" if use_cpu else f"cuda:{device_ids[0]}")

    # Auto-detect embedding dimension if requested
    dimension = args.dimension
    if dimension == -1:
        print("Auto-detecting embedding dimension from model...")
        dimension = auto_detect_dim(args.model, args.pretrained, device)
        print(f"Detected dimension: {dimension}")

    # Ensure collection exists (optionally recreate)
    ensure_collection(
        collection_name=args.collection_name,
        dimension=dimension,
        milvus_host=args.host,
        milvus_port=args.port,
        recreate=args.recreate,
        index_params={"metric_type": "COSINE", "index_type": "HNSW", "params": {"M": 32, "efConstruction": 512}},
    )

    # Discover files
    paths = discover_images(root, args.glob, args.start_index)
    if not paths:
        print("No files matched. Check --root and --glob.")
        sys.exit(0)

    # Enumerate with optional offset
    indexed_paths = list(enumerate(paths, start=args.id_offset))

    # Prepare global encode cache for CPU single-worker path
    if world_size == 1:
        _prepare_encode_cache(args.model, args.pretrained, device)
        encode_worker(
            rank=0,
            world_size=1,
            indexed_paths=indexed_paths,
            device_ids=device_ids,
            milvus_host=args.host,
            milvus_port=args.port,
            collection_name=args.collection_name,
            batch_size=args.batch_size,
            flush_interval=args.flush_interval,
            model_name=args.model,
            pretrained=args.pretrained,
            webp_field_names=("id", "filepath", "embedding", "video_id", "frame_id"),
        )
    else:
        # For multi-GPU, prep a tiny cache on rank 0 device so _flush_batch can access encode function
        # (each worker will create its own model too; this global cache is just for function binding)
        _prepare_encode_cache(args.model, args.pretrained, device)
        mp.set_start_method("spawn", force=True)
        mp.spawn(
            encode_worker,
            args=(
                world_size,
                indexed_paths,
                device_ids,
                args.host,
                args.port,
                args.collection_name,
                args.batch_size,
                args.flush_interval,
                args.model,
                args.pretrained,
                ("id", "filepath", "embedding", "video_id", "frame_id"),
            ),
            nprocs=world_size,
            join=True,
        )

    if args.build_index:
        build_index_and_load(
            collection_name=args.collection_name,
            index_params={"metric_type": "COSINE", "index_type": "HNSW", "params": {"M": 32, "efConstruction": 512}},
            milvus_host=args.host,
            milvus_port=args.port,
            field_name="embedding",
        )


if __name__ == "__main__":
    main()
