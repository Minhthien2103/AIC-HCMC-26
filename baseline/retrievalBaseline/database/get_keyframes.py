import argparse
import concurrent.futures
import csv
import glob
import os
from typing import List, Tuple

import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import open_clip


def preprocess_frame(frame, preprocess):
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return preprocess(pil_image).unsqueeze(0)  # (1, C, H, W)


@torch.no_grad()
def encode_batch(frames: List, model, preprocess, device: torch.device) -> torch.Tensor:
    # Parallelize PIL+preprocess on CPU
    with concurrent.futures.ThreadPoolExecutor() as ex:
        processed = list(ex.map(lambda fr: preprocess_frame(fr, preprocess), frames))
    images = torch.cat(processed, dim=0).to(device, non_blocking=True)  # (B, C, H, W)
    feats = model.encode_image(images)  # (B, D)
    feats = F.normalize(feats, dim=-1)  # unit-length for cosine similarity
    return feats


def save_image_webp(img_bgr, path: str, quality: int = 80, resize_factor: float = 0.5):
    if resize_factor != 1.0:
        img_bgr = cv2.resize(img_bgr, (0, 0), fx=resize_factor, fy=resize_factor)
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img_pil.save(path, format="WEBP", quality=quality)


def process_video(
    video_path: str,
    out_dir: str,
    maps_dir: str,
    model,
    preprocess,
    device: torch.device,
    clip_threshold: float,
    skip_frames: int,
    batch_size: int = 64,
    resize_factor: float = 0.5,
    webp_quality: int = 80,
) -> int:
    os.makedirs(out_dir, exist_ok=True)
    kf_dir = os.path.join(out_dir, "keyframes")
    os.makedirs(kf_dir, exist_ok=True)
    os.makedirs(maps_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    map_csv = os.path.join(maps_dir, f"{video_name}_map.csv")

    frames: List = []
    indices: List[int] = []
    keyframe_count = 0
    prev_feat = None

    with open(map_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["FrameID", "Seconds"])

        pbar = tqdm(total=total_frames if total_frames > 0 else None, desc=f"Processing {video_name}")
        frame_id = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # process every (skip_frames + 1)-th frame
            if skip_frames < 0 or (frame_id % (skip_frames + 1) == 0):
                frames.append(frame)
                indices.append(frame_id)

                if len(frames) >= batch_size:
                    feats = encode_batch(frames, model, preprocess, device)
                    for img, fid, feat in zip(frames, indices, feats):
                        if prev_feat is None:
                            # always take the very first considered frame
                            kf_path = os.path.join(kf_dir, f"keyframe_{fid}.webp")
                            save_image_webp(img, kf_path, quality=webp_quality, resize_factor=resize_factor)
                            writer.writerow([fid, f"{fid / fps:.2f}"])
                            prev_feat = feat
                            keyframe_count += 1
                        else:
                            sim = torch.dot(prev_feat, feat).item()  # cosine since normalized
                            if sim < clip_threshold:
                                kf_path = os.path.join(kf_dir, f"keyframe_{fid}.webp")
                                save_image_webp(img, kf_path, quality=webp_quality, resize_factor=resize_factor)
                                writer.writerow([fid, f"{fid / fps:.2f}"])
                                prev_feat = feat
                                keyframe_count += 1

                    frames.clear()
                    indices.clear()

            frame_id += 1
            pbar.update(1)

        # flush leftover
        if frames:
            feats = encode_batch(frames, model, preprocess, device)
            for img, fid, feat in zip(frames, indices, feats):
                if prev_feat is None:
                    kf_path = os.path.join(kf_dir, f"keyframe_{fid}.webp")
                    save_image_webp(img, kf_path, quality=webp_quality, resize_factor=resize_factor)
                    writer.writerow([fid, f"{fid / fps:.2f}"])
                    prev_feat = feat
                    keyframe_count += 1
                else:
                    sim = torch.dot(prev_feat, feat).item()
                    if sim < clip_threshold:
                        kf_path = os.path.join(kf_dir, f"keyframe_{fid}.webp")
                        save_image_webp(img, kf_path, quality=webp_quality, resize_factor=resize_factor)
                        writer.writerow([fid, f"{fid / fps:.2f}"])
                        prev_feat = feat
                        keyframe_count += 1

        pbar.close()
    cap.release()
    return keyframe_count


def process_all_videos(
    input_folder: str,
    output_base: str,
    clip_threshold: float,
    skip_frames: int,
    batch_size: int,
    resize_factor: float,
    webp_quality: int,
    start_index: int,
    pattern: str,
    device: torch.device,
):
    # Load OpenCLIP once
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14-378-quickgelu", pretrained="dfn5b"
    )
    model = model.to(device)
    model.eval()

    # discover videos
    video_files = sorted(glob.glob(os.path.join(input_folder, pattern)))
    if not video_files:
        print(f"No videos matched: {os.path.join(input_folder, pattern)}")
        return

    maps_dir = os.path.join(output_base, "maps")
    os.makedirs(maps_dir, exist_ok=True)

    for video_path in video_files[start_index:]:
        name = os.path.splitext(os.path.basename(video_path))[0]
        out_dir = os.path.join(output_base, name)
        os.makedirs(out_dir, exist_ok=True)

        print(f"Processing video: {name}")
        try:
            kf_count = process_video(
                video_path=video_path,
                out_dir=out_dir,
                maps_dir=maps_dir,
                model=model,
                preprocess=preprocess,
                device=device,
                clip_threshold=clip_threshold,
                skip_frames=skip_frames,
                batch_size=batch_size,
                resize_factor=resize_factor,
                webp_quality=webp_quality,
            )
            print(f"Total keyframes: {kf_count}")
        except Exception as e:
            print(f"Error processing {name}: {e}")
        print("-" * 20)


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract visual keyframes using OpenCLIP cosine similarity."
    )
    p.add_argument("--input-folder", type=str, required=True,
                   help="Folder containing videos.")
    p.add_argument("--output-base", type=str, default="./data-keyframes",
                   help="Base output folder (per-video subfolders + maps/).")
    p.add_argument("--pattern", type=str, default="*.mp4",
                   help="Glob pattern for videos (e.g., '*.mp4').")
    p.add_argument("--start-index", type=int, default=0,
                   help="Start processing from this sorted index.")
    p.add_argument("--clip-threshold", type=float, default=0.93,
                   help="Cosine similarity threshold to trigger a new keyframe (lower similarity => new keyframe).")
    p.add_argument("--skip-frames", type=int, default=5,
                   help="Process every (skip_frames + 1)-th frame. Use -1 to process every frame.")
    p.add_argument("--batch-size", type=int, default=64,
                   help="Batch size for CLIP encoding.")
    p.add_argument("--resize-factor", type=float, default=0.5,
                   help="Resize factor for saved WEBP keyframes (1.0 = original).")
    p.add_argument("--webp-quality", type=int, default=80,
                   help="WEBP quality (0-100).")
    p.add_argument("--cpu", action="store_true",
                   help="Force CPU even if CUDA is available.")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    process_all_videos(
        input_folder=args.input_folder,
        output_base=args.output_base,
        clip_threshold=args.clip_threshold,
        skip_frames=args.skip_frames,
        batch_size=args.batch_size,
        resize_factor=args.resize_factor,
        webp_quality=args.webp_quality,
        start_index=args.start_index,
        pattern=args.pattern,
        device=device,
    )


if __name__ == "__main__":
    main()
