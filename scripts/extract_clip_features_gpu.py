#!/usr/bin/env python3
"""High-speed GPU CLIP Feature Extractor for L01-L20.

Reads keyframe images directly from the local media server (http://localhost:8007)
over loopback, runs batch inference on NVIDIA GeForce RTX 4060 GPU,
and saves canonical L2-normalized float16 embeddings into data/clip-features-32/{video_id}.npy.

Zero Disk Waste:
- Images are streamed directly into RAM (as PIL/Tensors) and discarded after encoding.
- Only lightweight .npy feature arrays (~200KB per video) are written to disk.

Usage:
    python scripts/extract_clip_features_gpu.py [--prefixes L01,L02,...] [--batch-size 128] [--workers 24]
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src import config

try:
    import open_clip
except ImportError:
    print("[ERROR] open_clip is not installed in current environment.")
    sys.exit(1)


def fetch_and_preprocess(url: str, preprocess_fn, retries: int = 2) -> torch.Tensor | None:
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "AIC-Extractor/1.0"})
            with urllib.request.urlopen(req, timeout=4) as response:
                img = Image.open(io.BytesIO(response.read())).convert("RGB")
                return preprocess_fn(img)
        except Exception:
            if attempt < retries:
                time.sleep(0.1)
    return None


def process_video_features(
    video_id: str,
    csv_path: Path,
    out_path: Path,
    media_base: str,
    model,
    preprocess_fn,
    device: str,
    batch_size: int,
    workers: int,
) -> tuple[int, float, str]:
    start_t = time.time()

    # 1. Read CSV to get frame names
    rows: list[str] = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            n_str = str(r.get("n", "")).strip()
            if n_str.isdigit():
                rows.append(f"{int(n_str):03d}")

    total_frames = len(rows)
    if total_frames == 0:
        return 0, time.time() - start_t, "Empty CSV"

    # Check if existing npy matches exactly
    if out_path.is_file():
        try:
            existing = np.load(out_path, mmap_mode="r")
            if existing.shape == (total_frames, config.CLIP_DIM):
                return total_frames, time.time() - start_t, "Already exists (skipped)"
        except Exception:
            pass

    # 2. Process in batches with multithreaded image prefetching
    all_embeddings: list[np.ndarray] = []

    for i in range(0, total_frames, batch_size):
        batch_names = rows[i : i + batch_size]
        urls = [f"{media_base}/keyframes/{video_id}/{name}.jpg" for name in batch_names]

        # Fetch and transform in parallel
        with ThreadPoolExecutor(max_workers=min(workers, len(urls))) as executor:
            tensors = list(executor.map(lambda u: fetch_and_preprocess(u, preprocess_fn), urls))

        # Handle any rare missing frame with blank black frame
        valid_tensors = []
        blank_tensor = preprocess_fn(Image.new("RGB", (224, 224), color="black"))
        for t in tensors:
            valid_tensors.append(t if t is not None else blank_tensor)

        batch_tensor = torch.stack(valid_tensors).to(device)

        with torch.no_grad():
            features = model.encode_image(batch_tensor)
            # L2-normalize
            features = features / features.norm(dim=-1, keepdim=True)
            batch_np = features.cpu().numpy().astype(np.float16)
            all_embeddings.append(batch_np)

    # 3. Concatenate and save .npy
    full_array = np.vstack(all_embeddings)
    if full_array.shape != (total_frames, config.CLIP_DIM):
        return 0, time.time() - start_t, f"Shape mismatch: {full_array.shape} vs {(total_frames, config.CLIP_DIM)}"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, full_array)

    elapsed = time.time() - start_t
    return total_frames, elapsed, "OK"


def main():
    parser = argparse.ArgumentParser(description="Extract CLIP features using local RTX 4060 GPU.")
    parser.add_argument("--map-dir", type=str, default="data/map-keyframes")
    parser.add_argument("--out-dir", type=str, default="data/clip-features-32")
    parser.add_argument("--media-base", type=str, default="http://localhost:8007")
    parser.add_argument("--prefixes", type=str, default=",".join([f"L{i:02d}" for i in range(1, 21)]),
                        help="Comma-separated prefixes to process (e.g. L01,L02... or ALL)")
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size")
    parser.add_argument("--workers", type=int, default=24, help="Threads for network image downloading")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70, flush=True)
    print(" AIC GPU Feature Extraction (OpenCLIP ViT-B-32-quickgelu)", flush=True)
    print("=" * 70, flush=True)
    print(f" Computing Device: {device.upper()}", flush=True)
    if device == "cuda":
        print(f" GPU Name:         {torch.cuda.get_device_name(0)}", flush=True)
    print(f" Media Endpoint:   {args.media_base}", flush=True)
    print(f" Batch Size:       {args.batch_size}", flush=True)
    print(f" Network Workers:  {args.workers}", flush=True)
    print("=" * 70, flush=True)

    # Load Model
    print(f"Loading {config.CLIP_MODEL_NAME} ({config.CLIP_PRETRAINED}) onto {device}...", flush=True)
    model, _, preprocess = open_clip.create_model_and_transforms(
        config.CLIP_MODEL_NAME,
        pretrained=config.CLIP_PRETRAINED,
        device=device
    )
    model.eval()

    map_dir = Path(args.map_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(map_dir.glob("*.csv"))
    if args.prefixes.strip().upper() != "ALL":
        prefix_set = {p.strip().upper() for p in args.prefixes.split(",") if p.strip()}
        csv_files = [f for f in csv_files if f.stem.split("_")[0].upper() in prefix_set]

    print(f"Total Videos to Process: {len(csv_files)}", flush=True)
    print("-" * 70, flush=True)

    t0 = time.time()
    success_count = 0
    total_frames_extracted = 0

    for idx, csv_path in enumerate(csv_files, 1):
        video_id = csv_path.stem
        out_path = out_dir / f"{video_id}.npy"

        count, elapsed, msg = process_video_features(
            video_id=video_id,
            csv_path=csv_path,
            out_path=out_path,
            media_base=args.media_base,
            model=model,
            preprocess_fn=preprocess,
            device=device,
            batch_size=args.batch_size,
            workers=args.workers,
        )

        if msg == "OK" or "skipped" in msg:
            success_count += 1
            total_frames_extracted += count
            status = "✓"
        else:
            status = "✗"

        pct = (idx / len(csv_files)) * 100
        speed = count / elapsed if elapsed > 0 else 0
        print(f"[{pct:5.1f}%] [{idx:3d}/{len(csv_files):3d}] {status} {video_id:10s} -> {count:4d} frames ({elapsed:.1f}s, {speed:4.1f} fps) - {msg}", flush=True)

    total_time = time.time() - t0
    print("=" * 70, flush=True)
    print(f" Feature Extraction Finished in {total_time:.1f}s ({total_time / 60:.2f} min)", flush=True)
    print(f" Success: {success_count}/{len(csv_files)} videos", flush=True)
    print(f" Total Embeddings: {total_frames_extracted}", flush=True)
    print(f" Output Location:  {out_dir}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
