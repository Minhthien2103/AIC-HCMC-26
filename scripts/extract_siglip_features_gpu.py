#!/usr/bin/env python3
"""High-speed GPU SigLIP-SO400M Feature Extractor for AIC 2026 (L01-L24).

Model: ViT-SO400M-14-SigLIP (Pretrained: webli)
Embedding Dimension: 1152
Output Directory: data/siglip-features-so400m/{video_id}.npy

Architecture:
- Streams keyframe images directly from local media server (http://localhost:8007) into RAM.
- Encodes in batches on NVIDIA GeForce RTX 4060 GPU with FP16 autocast.
- Saves L2-normalized float16 embeddings.
- Zero disk waste, supports pause & resume gracefully.
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

try:
    import open_clip
except ImportError:
    print("[ERROR] open_clip is not installed in current environment.")
    sys.exit(1)

SIGLIP_MODEL_NAME = "ViT-SO400M-14-SigLIP"
SIGLIP_PRETRAINED = "webli"
SIGLIP_DIM = 1152


def fetch_and_preprocess(url: str, preprocess_fn, retries: int = 2) -> torch.Tensor | None:
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "AIC-SigLIP-Extractor/1.0"})
            with urllib.request.urlopen(req, timeout=4) as response:
                img = Image.open(io.BytesIO(response.read())).convert("RGB")
                return preprocess_fn(img)
        except Exception:
            if attempt < retries:
                time.sleep(0.08)
    return None


def process_video_siglip(
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

    # 1. Read CSV to get keyframe names
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
            if existing.shape == (total_frames, SIGLIP_DIM):
                return total_frames, time.time() - start_t, "Skipped (Already exists)"
        except Exception:
            pass

    # 2. Batch inference
    all_embeddings: list[np.ndarray] = []
    blank_tensor = preprocess_fn(Image.new("RGB", (224, 224), color="black"))

    for i in range(0, total_frames, batch_size):
        batch_names = rows[i : i + batch_size]
        urls = [f"{media_base}/keyframes/{video_id}/{name}.jpg" for name in batch_names]

        # Parallel network fetching
        with ThreadPoolExecutor(max_workers=min(workers, len(urls))) as executor:
            tensors = list(executor.map(lambda u: fetch_and_preprocess(u, preprocess_fn), urls))

        valid_tensors = [t if t is not None else blank_tensor for t in tensors]
        batch_tensor = torch.stack(valid_tensors).to(device)

        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=(device == "cuda")):
            features = model.encode_image(batch_tensor)
            # L2-normalize
            features = features / features.norm(dim=-1, keepdim=True)
            batch_np = features.cpu().float().numpy().astype(np.float16)
            all_embeddings.append(batch_np)

    # 3. Save .npy
    full_array = np.vstack(all_embeddings)
    if full_array.shape != (total_frames, SIGLIP_DIM):
        return 0, time.time() - start_t, f"Shape mismatch: {full_array.shape} vs {(total_frames, SIGLIP_DIM)}"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, full_array)

    elapsed = time.time() - start_t
    return total_frames, elapsed, "OK"


def main():
    parser = argparse.ArgumentParser(description="Extract SigLIP-SO400M features using local RTX 4060 GPU.")
    parser.add_argument("--map-dir", type=str, default="data/map-keyframes")
    parser.add_argument("--out-dir", type=str, default="data/siglip-features-so400m")
    parser.add_argument("--media-base", type=str, default="http://localhost:8007")
    parser.add_argument("--prefixes", type=str, default=",".join([f"L{i:02d}" for i in range(1, 25)]),
                        help="Comma-separated prefixes to process (e.g. L01,L02... or ALL)")
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size (64 fits easily in 8GB VRAM)")
    parser.add_argument("--workers", type=int, default=24, help="Threads for network image downloading")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    map_dir = Path(args.map_dir)
    out_dir = Path(args.out_dir)

    print("=" * 72, flush=True)
    print(" AIC GPU Feature Extractor: Google SigLIP-SO400M (1152-dim)", flush=True)
    print("=" * 72, flush=True)
    print(f" Computing Device: {device.upper()}", flush=True)
    if device == "cuda":
        print(f" GPU Name:         {torch.cuda.get_device_name(0)} (8GB VRAM)", flush=True)
    print(f" Model:            {SIGLIP_MODEL_NAME} ({SIGLIP_PRETRAINED})", flush=True)
    print(f" Embedding Dim:    {SIGLIP_DIM}", flush=True)
    print(f" Output Directory: {out_dir}", flush=True)
    print(f" Media Endpoint:   {args.media_base}", flush=True)
    print(f" Batch Size:       {args.batch_size}", flush=True)
    print(f" Network Workers:  {args.workers}", flush=True)
    print("=" * 72, flush=True)

    # Check media endpoint connectivity
    print(f"Testing connection to {args.media_base}...", flush=True)
    try:
        req = urllib.request.Request(f"{args.media_base}/", headers={"User-Agent": "Test/1.0"})
        with urllib.request.urlopen(req, timeout=3) as res:
            print(f"✓ Media server is ONLINE (HTTP {res.status})", flush=True)
    except Exception as e:
        print(f"⚠️ CẢNH BÁO: Không thể kết nối tới {args.media_base} ({e}).", flush=True)
        print("  Hãy đảm bảo bạn đã kết nối OpenVPN và bật SSH Tunnel trước:")
        print("  ssh -N -L 8007:localhost:8007 team12@192.168.28.112\n", flush=True)

    # Load Model
    print(f"Loading {SIGLIP_MODEL_NAME} onto {device}...", flush=True)
    load_start = time.time()
    model, _, preprocess = open_clip.create_model_and_transforms(
        SIGLIP_MODEL_NAME,
        pretrained=SIGLIP_PRETRAINED,
        device=device
    )
    model.eval()
    print(f"✓ Model loaded successfully in {time.time() - load_start:.1f}s\n", flush=True)

    # Filter target videos (L01 through L24)
    target_prefixes = set(args.prefixes.split(",")) if args.prefixes != "ALL" else None
    csv_files: list[Path] = []
    for csv_file in sorted(map_dir.glob("*.csv")):
        vid = csv_file.stem
        prefix = vid.split("_")[0]
        if target_prefixes is None or prefix in target_prefixes:
            csv_files.append(csv_file)

    total_videos = len(csv_files)
    print(f"Found {total_videos} videos to extract for {args.prefixes}\n", flush=True)

    total_frames_extracted = 0
    start_total_time = time.time()

    for idx, csv_path in enumerate(csv_files, 1):
        vid = csv_path.stem
        out_path = out_dir / f"{vid}.npy"

        frames, elapsed, status = process_video_siglip(
            video_id=vid,
            csv_path=csv_path,
            out_path=out_path,
            media_base=args.media_base,
            model=model,
            preprocess_fn=preprocess,
            device=device,
            batch_size=args.batch_size,
            workers=args.workers,
        )

        total_frames_extracted += frames
        pct = (idx / total_videos) * 100
        fps = (frames / elapsed) if elapsed > 0 else 0.0

        if "Skipped" in status:
            print(f"[{pct:5.1f}%] [{idx:3d}/{total_videos}] ⏩ {vid:<10} -> {status}", flush=True)
        elif status == "OK":
            print(f"[{pct:5.1f}%] [{idx:3d}/{total_videos}] ✓ {vid:<10} -> {frames:4d} frames ({elapsed:5.1f}s | {fps:5.1f} fps) - OK", flush=True)
        else:
            print(f"[{pct:5.1f}%] [{idx:3d}/{total_videos}] ❌ {vid:<10} -> {status}", flush=True)

    total_time = time.time() - start_total_time
    print("=" * 72, flush=True)
    print(f" Extraction Completed in {total_time:.1f}s ({total_time / 60:.2f} min)", flush=True)
    print(f" Total Frames Encoded: {total_frames_extracted}", flush=True)
    print(f" Output Directory:     {out_dir.resolve()}", flush=True)
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
