#!/usr/bin/env python3
"""Automated Keyframe & Map-Keyframe Extractor for L01-L20 on Ubuntu Server.

Extracts keyframes at 1 frame per second (1 FPS) uniformly from MP4 videos,
saves JPEG images into ~/aic-data/keyframes/{video_id}/{n:03d}.jpg,
and generates canonical CSVs at ~/aic-data/map-keyframes/{video_id}.csv.

Features:
- High speed multi-processing with cv2.grab() (skips non-keyframe decoding).
- Auto-generates map-keyframes CSV matching AIC schema: (n, pts_time, fps, frame_idx).
- Resumable: automatically skips already processed videos.
- Flushed stdout for real-time tail -f monitoring.

Usage:
    python3 extract_keyframes_l01_l20.py [--workers 11] [--fps 1.0]
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
    print("[WARN] Installing opencv-python-headless...", flush=True)
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "opencv-python-headless"])
    import cv2


def get_target_videos(videos_dirs: list[Path], target_prefixes: set[str]) -> list[Path]:
    """Find all MP4 files matching target prefixes (e.g. L01 to L20)."""
    found: dict[str, Path] = {}
    for base in videos_dirs:
        if not base.is_dir():
            continue
        for p in base.glob("*.mp4"):
            vid = p.stem.upper()
            prefix = vid.split("_")[0]
            if prefix in target_prefixes and vid not in found:
                found[vid] = p
    return sorted(found.values(), key=lambda p: p.stem)


def process_video(args: tuple[Path, Path, Path, float]) -> tuple[str, int, float, str]:
    video_path, keyframes_base_dir, map_base_dir, sample_fps = args
    video_id = video_path.stem
    start_t = time.time()

    try:
        csv_path = map_base_dir / f"{video_id}.csv"
        video_out_dir = keyframes_base_dir / video_id

        # Check if already completed (resumable)
        if csv_path.is_file() and video_out_dir.is_dir():
            jpg_count = len(list(video_out_dir.glob("*.jpg")))
            if jpg_count > 0:
                return video_id, jpg_count, time.time() - start_t, "Already extracted (skipped)"

        video_out_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return video_id, 0, time.time() - start_t, "Cannot open video"

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 240:
            fps = 30.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, int(round(fps / sample_fps)))

        target_indices = list(range(0, total_frames, step))
        if not target_indices:
            target_indices = [0]

        num_targets = len(target_indices)
        csv_rows: list[dict] = []
        target_idx = 0
        cur_frame = 0
        saved_count = 0

        while target_idx < num_targets:
            grabbed = cap.grab()
            if not grabbed:
                break

            if cur_frame >= target_indices[target_idx]:
                ret, frame = cap.retrieve()
                if ret and frame is not None:
                    n = target_idx + 1
                    k_name = f"{n:03d}"
                    out_path = video_out_dir / f"{k_name}.jpg"
                    cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    saved_count += 1

                    pts_time = round(cur_frame / fps, 4)
                    csv_rows.append({
                        "n": n,
                        "pts_time": pts_time,
                        "fps": round(fps, 2),
                        "frame_idx": cur_frame
                    })

                target_idx += 1
                while target_idx < num_targets and cur_frame >= target_indices[target_idx]:
                    target_idx += 1

            cur_frame += 1

        cap.release()

        # Save CSV
        if csv_rows:
            with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["n", "pts_time", "fps", "frame_idx"])
                writer.writeheader()
                writer.writerows(csv_rows)

        elapsed = time.time() - start_t
        return video_id, saved_count, elapsed, "OK"
    except Exception as e:
        return video_id, 0, time.time() - start_t, f"Error: {e}"


def main():
    parser = argparse.ArgumentParser(description="Extract keyframes and map CSVs for L01-L20 on server.")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames per second to extract (default: 1.0)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1),
                        help="Number of CPU worker processes")
    parser.add_argument("--out-keyframes", type=str, default="/home/team12/aic-data/keyframes")
    parser.add_argument("--out-map", type=str, default="/home/team12/aic-data/map-keyframes")
    args = parser.parse_args()

    # Prefixes L01 to L20
    target_prefixes = {f"L{i:02d}" for i in range(1, 21)}

    search_dirs = [
        Path("/home/team12/aic-data/videos"),
        Path("/home/team12/aic-data/mlcv1/Datasets/HCMAI24/updated/videos/batch1"),
        Path("/home/team12/aic-data/mlcv1/Datasets/HCMAI24/updated/videos/batch2"),
    ]

    kf_dir = Path(args.out_keyframes).expanduser()
    map_dir = Path(args.out_map).expanduser()
    kf_dir.mkdir(parents=True, exist_ok=True)
    map_dir.mkdir(parents=True, exist_ok=True)

    videos = get_target_videos(search_dirs, target_prefixes)

    print("=" * 68, flush=True)
    print(" AIC Pipeline: Automated L01-L20 Keyframe & Map Extractor", flush=True)
    print("=" * 68, flush=True)
    print(f" Target Groups:    L01 through L20 ({len(target_prefixes)} groups)", flush=True)
    print(f" Total Videos:     {len(videos)}", flush=True)
    print(f" Sampling Rate:    {args.fps} keyframe per second (1 FPS)", flush=True)
    print(f" CPU Workers:      {args.workers}", flush=True)
    print(f" Output Keyframes: {kf_dir}", flush=True)
    print(f" Output Map CSVs:  {map_dir}", flush=True)
    print("=" * 68, flush=True)

    if not videos:
        print("[ERROR] No L01-L20 videos found in search directories!", flush=True)
        sys.exit(1)

    tasks = [(p, kf_dir, map_dir, args.fps) for p in videos]

    t0 = time.time()
    success_count = 0
    total_frames = 0

    with mp.Pool(processes=args.workers) as pool:
        for idx, (v_id, count, elapsed, msg) in enumerate(pool.imap_unordered(process_video, tasks), 1):
            if msg == "OK" or "skipped" in msg:
                success_count += 1
                total_frames += count
                status_icon = "✓"
            else:
                status_icon = "✗"

            pct = (idx / len(tasks)) * 100
            print(f"[{pct:5.1f}%] [{idx:3d}/{len(tasks):3d}] {status_icon} {v_id:10s} -> {count:3d} frames ({elapsed:.1f}s) - {msg}", flush=True)

    total_time = time.time() - t0
    print("=" * 68, flush=True)
    print(f" Completed in:          {total_time:.1f}s ({total_time / 60:.2f} min)", flush=True)
    print(f" Success Videos:        {success_count}/{len(tasks)}", flush=True)
    print(f" Total Keyframes Saved: {total_frames}", flush=True)
    print(f" Output Keyframes:      {kf_dir}", flush=True)
    print(f" Output Map CSVs:       {map_dir}", flush=True)
    print("=" * 68, flush=True)


if __name__ == "__main__":
    main()
