#!/usr/bin/env python3
"""High-speed Multi-process Keyframe Extractor for Ubuntu Server.

Extracts keyframe JPEG images from MP4 videos based on map-keyframes CSV files.
Optimized for multi-core CPUs using OpenCV sequential packet grabbing (cap.grab)
which avoids decoding non-keyframe video frames.

Usage:
    python3 extract_keyframes_server.py [--prefix L21,L22,L23,L24] [--workers 8]
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

try:
    import cv2
except ImportError:
    print("[WARN] opencv-python is not installed. Installing opencv-python-headless...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "opencv-python-headless"])
    import cv2


def find_video_path(video_id: str, search_dirs: list[Path]) -> Path | None:
    for base in search_dirs:
        candidate = base / f"{video_id}.mp4"
        if candidate.is_file():
            return candidate
    return None


def process_single_video(args: tuple[str, Path, list[Path], Path]) -> tuple[str, int, float, str]:
    video_id, csv_path, search_dirs, out_base_dir = args
    start_t = time.time()

    # 1. Read map-keyframes CSV
    targets: list[tuple[int, str]] = []  # (frame_idx, "001")
    try:
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                n_str = str(row.get("n", "")).strip()
                frame_idx_str = str(row.get("frame_idx", "")).strip()
                if n_str.isdigit() and frame_idx_str.isdigit():
                    k_name = n_str.zfill(3)
                    targets.append((int(frame_idx_str), k_name))
    except Exception as e:
        return video_id, 0, time.time() - start_t, f"CSV Error: {e}"

    if not targets:
        return video_id, 0, time.time() - start_t, "Empty CSV"

    # Sort targets by frame index
    targets.sort(key=lambda x: x[0])

    # 2. Prepare output directory
    video_out_dir = out_base_dir / video_id
    video_out_dir.mkdir(parents=True, exist_ok=True)

    # Check if already fully extracted
    existing = {p.stem for p in video_out_dir.glob("*.jpg")}
    all_target_names = {t[1] for t in targets}
    if all_target_names.issubset(existing):
        return video_id, len(targets), time.time() - start_t, "Already extracted (skipped)"

    # 3. Locate MP4 video
    video_path = find_video_path(video_id, search_dirs)
    if not video_path:
        return video_id, 0, time.time() - start_t, "Video MP4 file not found"

    # 4. Extract frames using fast sequential grab
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return video_id, 0, time.time() - start_t, "Cannot open video"

    target_idx = 0
    cur_frame = 0
    saved_count = 0
    num_targets = len(targets)

    while target_idx < num_targets:
        grabbed = cap.grab()
        if not grabbed:
            break

        if cur_frame >= targets[target_idx][0]:
            ret, frame = cap.retrieve()
            if ret and frame is not None:
                out_path = video_out_dir / f"{targets[target_idx][1]}.jpg"
                cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                saved_count += 1

            target_idx += 1
            # Handle any skipped targets with the same decoded frame
            while target_idx < num_targets and cur_frame >= targets[target_idx][0]:
                if ret and frame is not None:
                    out_path = video_out_dir / f"{targets[target_idx][1]}.jpg"
                    cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    saved_count += 1
                target_idx += 1

        cur_frame += 1

    cap.release()
    elapsed = time.time() - start_t
    return video_id, saved_count, elapsed, "OK"


def main():
    parser = argparse.ArgumentParser(description="Extract keyframes from videos on Ubuntu server.")
    parser.add_argument("--map-dir", type=str, default="/home/team12/aic-data/map-keyframes",
                        help="Path to map-keyframes folder containing CSV files")
    parser.add_argument("--out-dir", type=str, default="/home/team12/aic-data/keyframes",
                        help="Path to output keyframes folder")
    parser.add_argument("--prefixes", type=str, default="L21,L22,L23,L24",
                        help="Comma-separated prefixes to process (e.g. L21,L22,L23,L24 or ALL)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1),
                        help="Number of parallel worker processes")
    args = parser.parse_args()

    map_dir = Path(args.map_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    search_dirs = [
        Path("/home/team12/aic-data/videos"),
        Path("/home/team12/aic-data/mlcv1/Datasets/HCMAI24/updated/videos/batch2"),
        Path("/home/team12/aic-data/mlcv1/Datasets/HCMAI24/updated/videos/batch1"),
    ]

    if not map_dir.is_dir():
        print(f"[ERROR] Map directory not found: {map_dir}")
        print("Please copy map-keyframes from local PC to server first:")
        print("  scp -r data/map-keyframes team12@192.168.28.112:~/aic-data/")
        sys.exit(1)

    csv_files = sorted(map_dir.glob("*.csv"))
    if args.prefixes.strip().upper() != "ALL":
        prefix_set = {p.strip().upper() for p in args.prefixes.split(",") if p.strip()}
        csv_files = [f for f in csv_files if f.stem.split("_")[0].upper() in prefix_set]

    print("=" * 65)
    print(f" AIC Keyframe Extractor - Multi-core CPU Processing")
    print("=" * 65)
    print(f"  Target Prefixes: {args.prefixes}")
    print(f"  Total Videos:    {len(csv_files)}")
    print(f"  CPU Workers:     {args.workers}")
    print(f"  Map Directory:   {map_dir}")
    print(f"  Output Keyframes:{out_dir}")
    print("=" * 65)

    if not csv_files:
        print("[WARN] No matching CSV files found. Check prefix filter or map directory.")
        sys.exit(0)

    tasks = [(f.stem, f, search_dirs, out_dir) for f in csv_files]

    total_saved = 0
    success_count = 0
    t0 = time.time()

    with mp.Pool(processes=args.workers) as pool:
        for idx, (v_id, count, elapsed, msg) in enumerate(pool.imap_unordered(process_single_video, tasks), 1):
            if msg == "OK" or "skipped" in msg:
                success_count += 1
                total_saved += count
                status_icon = "✓"
            else:
                status_icon = "✗"

            progress = idx / len(tasks) * 100
            print(f"[{progress:5.1f}%] [{idx:3d}/{len(tasks):3d}] {status_icon} {v_id:10s} -> {count:3d} frames ({elapsed:.1f}s) - {msg}")

    total_time = time.time() - t0
    print("=" * 65)
    print(f" Extraction Completed in {total_time:.1f}s ({total_time/60:.2f} min)")
    print(f" Success Videos: {success_count}/{len(tasks)}")
    print(f" Total Keyframes Saved: {total_saved}")
    print(f" Output Location: {out_dir}")
    print("=" * 65)


if __name__ == "__main__":
    main()
