#!/usr/bin/env python3
"""High-speed Subsampled EasyOCR Extractor for AIC-HCMC 2026.

Features:
- Subsamples 1 frame every 4 seconds (stride = 4) to finish within ~8-10 hours.
- Streams keyframes directly from media server (http://localhost:8007) via SSH Tunnel.
- Saves per-video checkpoints into data/ocr_cache/{video_id}.json (instant resume if interrupted).
- Generates final compiled indexes/ocr_text.parquet.

Usage:
  python scripts/extract_ocr_easyocr.py [--media-base http://localhost:8007] [--stride 4] [--prefixes L01,L02... or ALL]
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src import config

try:
    import easyocr
except ImportError:
    print("[ERROR] easyocr is not installed. Run: pip install easyocr")
    sys.exit(1)


def fetch_image_bytes(url: str, retries: int = 2) -> bytes | None:
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "AIC-OCR-Extractor/1.0"})
            with urllib.request.urlopen(req, timeout=5) as response:
                return response.read()
        except Exception:
            if attempt < retries:
                time.sleep(0.1)
    return None


def process_video_ocr(
    video_id: str,
    frames: list[dict],
    reader: easyocr.Reader,
    cache_dir: Path,
    media_base: str,
    stride: int = 4,
    workers: int = 4,
) -> tuple[int, int, float, str]:
    """Process a single video: subsample frames, run EasyOCR, save JSON cache."""
    start_t = time.time()
    cache_file = cache_dir / f"{video_id}.json"

    # 1. Check if already processed (Resume support)
    if cache_file.is_file():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return len(frames), len(data), time.time() - start_t, "Skipped (Already cached)"
        except Exception:
            pass

    # 2. Subsample frames (every `stride` seconds)
    # Sort frames by pts_time
    sorted_frames = sorted(frames, key=lambda x: x["pts_time"])
    subsampled = sorted_frames[::stride]
    if not subsampled:
        return len(frames), 0, time.time() - start_t, "No frames"

    # 3. Stream images in parallel batches
    urls = [f"{media_base}/keyframes/{video_id}/{f['keyframe_name']}.jpg" for f in subsampled]
    with ThreadPoolExecutor(max_workers=min(workers, len(urls))) as executor:
        raw_bytes_list = list(executor.map(fetch_image_bytes, urls))

    records: list[dict] = []
    text_count = 0

    # 4. Run EasyOCR inference on GPU
    for f_meta, raw_bytes in zip(subsampled, raw_bytes_list):
        if raw_bytes is None:
            continue
        try:
            img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            img_np = np.array(img)
            # detail=1 returns list of [bbox, text, prob]
            results = reader.readtext(img_np, detail=1, paragraph=False)

            detected_texts = []
            for bbox, text, prob in results:
                clean_text = text.strip()
                if clean_text and prob >= 0.35 and len(clean_text) >= 2:
                    detected_texts.append({
                        "text": clean_text,
                        "prob": round(float(prob), 3)
                    })

            if detected_texts:
                # Concatenate all detected texts into a single searchable string
                full_text = " | ".join([t["text"] for t in detected_texts])
                avg_prob = float(np.mean([t["prob"] for t in detected_texts]))
                records.append({
                    "video_id": video_id,
                    "keyframe_name": f_meta["keyframe_name"],
                    "frame_id": f_meta["frame_id"],
                    "pts_time": f_meta["pts_time"],
                    "ocr_text": full_text,
                    "details": detected_texts,
                    "conf": round(avg_prob, 3)
                })
                text_count += 1
        except Exception as e:
            continue

    # 5. Save per-video JSON checkpoint
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - start_t
    return len(subsampled), text_count, elapsed, "OK"


def compile_final_parquet(cache_dir: Path, output_parquet: Path):
    """Compile all per-video JSON caches into one master Parquet table."""
    print("\n" + "=" * 65)
    print(">>> Compiling all OCR checkpoints into Parquet...")
    json_files = sorted(list(cache_dir.glob("*.json")))
    if not json_files:
        print("[WARN] No JSON files found to compile.")
        return

    all_rows = []
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                items = json.load(f)
                for item in items:
                    all_rows.append({
                        "video_id": item["video_id"],
                        "keyframe_name": str(item["keyframe_name"]),
                        "frame_id": int(item["frame_id"]),
                        "pts_time": float(item["pts_time"]),
                        "ocr_text": str(item["ocr_text"]),
                        "conf": float(item.get("conf", 1.0))
                    })
        except Exception:
            continue

    if all_rows:
        df = pl.DataFrame(all_rows)
        output_parquet.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(output_parquet)
        print(f"Successfully compiled {len(df):,} OCR records across {len(json_files)} videos!")
        print(f"Saved to: {output_parquet}")
    else:
        print("[WARN] No valid OCR text records found.")


def main():
    parser = argparse.ArgumentParser(description="Subsampled EasyOCR Extractor for AIC 2026.")
    parser.add_argument("--media-base", type=str, default="http://localhost:8007", help="Base URL of media server")
    parser.add_argument("--stride", type=int, default=4, help="Subsample step in frames (default: 4s/frame)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel download workers")
    parser.add_argument("--prefixes", type=str, default="ALL", help="Video batches to process, e.g. L01,L02 or ALL")
    parser.add_argument("--compile-only", action="store_true", help="Compile existing cache into Parquet without running OCR")
    args = parser.parse_args()

    cache_dir = config.DATA_DIR / "ocr_cache"
    out_parquet = config.INDEX_DIR / "ocr_text.parquet"

    if args.compile_only:
        compile_final_parquet(cache_dir, out_parquet)
        return

    # Check media server reachability
    test_url = f"{args.media_base}/keyframes/L01_V001/001.jpg"
    print(f"Testing media server connection: {test_url} ...")
    try:
        req = urllib.request.Request(test_url, headers={"User-Agent": "Test"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            print(f"Media server reachable! Status: {resp.status}\n")
    except Exception as e:
        print(f"[ERROR] Cannot connect to media server at {args.media_base}: {e}")
        print("  Hãy đảm bảo bạn đã kết nối OpenVPN và bật SSH Tunnel:")
        print("  ssh -N -L 8007:localhost:8007 team12@192.168.28.112\n")
        sys.exit(1)

    # Load master metadata to group by video
    print(f"Loading metadata from {config.METADATA_PATH} ...")
    meta_df = pl.read_parquet(config.METADATA_PATH)
    all_videos = sorted(meta_df["video_id"].unique().to_list())

    # Filter by prefixes
    if args.prefixes != "ALL":
        target_prefixes = [p.strip().upper() for p in args.prefixes.split(",") if p.strip()]
        all_videos = [v for v in all_videos if any(v.startswith(p) for p in target_prefixes)]

    print(f"Total videos to process: {len(all_videos)} (Stride = {args.stride}s)")

    # Initialize EasyOCR
    print("\nInitializing EasyOCR Reader on GPU (CUDA)...")
    reader = easyocr.Reader(["vi", "en"], gpu=True)
    print("EasyOCR Reader ready!\n")

    # Process videos sequentially, images inside video downloaded concurrently
    total_start = time.time()
    total_processed_frames = 0
    total_text_frames = 0

    for idx, video_id in enumerate(all_videos, 1):
        # Extract frames metadata for this video
        vid_rows = meta_df.filter(pl.col("video_id") == video_id).select(
            ["keyframe_name", "frame_id", "pts_time"]
        ).to_dicts()

        n_sub, n_text, elapsed, status = process_video_ocr(
            video_id=video_id,
            frames=vid_rows,
            reader=reader,
            cache_dir=cache_dir,
            media_base=args.media_base,
            stride=args.stride,
            workers=args.workers,
        )

        total_processed_frames += n_sub
        total_text_frames += n_text

        speed = n_sub / elapsed if elapsed > 0 else 0
        print(f"[{idx:03d}/{len(all_videos):03d}] {video_id:<10}: {n_sub:4d} frames | {n_text:3d} with text | {elapsed:5.1f}s ({speed:4.1f} fps) -> {status}")

    total_time = time.time() - total_start
    print("\n" + "=" * 65)
    print(f"Done processing {len(all_videos)} videos in {total_time/3600:.2f} hours!")
    print(f"Total frames scanned: {total_processed_frames:,} | Frames with text: {total_text_frames:,}")

    # Compile final parquet
    compile_final_parquet(cache_dir, out_parquet)


if __name__ == "__main__":
    main()
