"""
Stage 4 - Keyframe Captioning via Gemini API / Local VLM
Input:  data/keyframes/{video_id}/{n}.jpg
        data/map-keyframes/{video_id}.csv   (frame_idx, pts_time)
        transcripts.parquet                 (video_id, start_time, end_time, text)  [optional]
Output: data/captions.parquet              (video_id, keyframe_name, frame_idx, pts_time, shot_id, caption)
"""
import os, sys, json, time
from pathlib import Path
import pandas as pd
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
from src import config

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = genai.GenerativeModel("gemini-3.1-pro") # User requested Gemini 3.1 Pro

CAPTIONS_PATH = config.DATA_DIR / "captions.parquet"
TRANSCRIPTS_PATH = config.DATA_DIR / "transcripts.parquet"
CONTEXT_K = 2  # Surrounding frames to include as context


def _load_transcripts() -> dict[str, list[dict]]:
    """Load transcripts indexed by video_id -> list of {start, end, text}."""
    if not TRANSCRIPTS_PATH.exists():
        return {}
    df = pd.read_parquet(TRANSCRIPTS_PATH)
    result = {}
    for row in df.itertuples():
        vid = row.video_id
        result.setdefault(vid, []).append({
            "start": float(row.start_time),
            "end":   float(row.end_time),
            "text":  str(row.text),
        })
    return result


def _get_transcript_for_time(transcripts: list[dict], pts: float, window: float = 5.0) -> str:
    """Retrieve transcript segments within ±window seconds of pts."""
    segs = [s["text"] for s in transcripts if abs(s["start"] - pts) <= window]
    return " ".join(segs).strip()


def _caption_keyframe(
    image_path: Path,
    local_vlm,
    prev_imgs: list[Path],
    next_imgs: list[Path],
    transcript: str,
) -> str:
    parts = []
    # Context frames first
    for p in prev_imgs:
        parts.append(Image.open(p).convert("RGB"))
    parts.append(Image.open(image_path).convert("RGB"))
    for n in next_imgs:
        parts.append(Image.open(n).convert("RGB"))

    ctx_note = f'\nAudio transcript near this moment: "{transcript}"' if transcript else ""

    prompt = (
        "You are analyzing a sequence of video keyframes. "
        f"The MIDDLE image is the TARGET frame. The others are temporal context.{ctx_note}\n\n"
        "Generate ONE precise English caption for the TARGET frame. "
        "Include: (1) main subjects and their clothing/attributes, "
        "(2) action or event happening, (3) setting/location. "
        "Be concise (max 2 sentences). Output ONLY the caption text."
    )
    parts.append(prompt)

    try:
        if local_vlm:
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            with Image.open(image_path) as img:
                img_rgb = img.convert("RGB")
            return local_vlm._generate_text(local_vlm._prepare(messages, img_rgb), max_new_tokens=128)
        else:
            response = MODEL.generate_content(parts)
            return response.text.strip()
    except Exception as e:
        print(f"  [Caption ERROR] {e}")
        return ""


def run_stage4(video_ids: list[str] | None = None, resume: bool = True, use_local: bool = False):
    if use_local:
        from src.online_pipeline.vlm_pipeline import VLMPipeline
        local_vlm = VLMPipeline()
        local_vlm.load()
    else:
        local_vlm = None

    all_transcripts = _load_transcripts()
    existing = set()
    rows = []

    if resume and CAPTIONS_PATH.exists():
        df_exist = pd.read_parquet(CAPTIONS_PATH)
        rows = df_exist.to_dict("records")
        existing = set(zip(df_exist.video_id, df_exist.keyframe_name))
        print(f"[Stage4] Resuming: {len(existing)} already captioned.")

    if video_ids is None:
        video_ids = sorted(p.name for p in config.KEYFRAMES_DIR.iterdir() if p.is_dir())

    for vid in video_ids:
        map_path = config.MAP_KEYFRAMES_DIR / f"{vid}.csv"
        if not map_path.exists():
            print(f"[Stage4] No map-keyframes for {vid}, skipping.")
            continue

        df_map = pd.read_csv(map_path)
        keyframes = sorted(config.KEYFRAMES_DIR / vid, key=lambda p: p.name)
        transcripts = all_transcripts.get(vid, [])

        for i, kf_path in enumerate(keyframes):
            kf_name = kf_path.stem  # e.g. "001"
            if (vid, kf_name) in existing:
                continue

            # Get frame metadata
            idx = i
            pts = float(df_map.iloc[i]["pts_time"]) if i < len(df_map) else 0.0
            shot_id = 0  # Will be filled by shot detection if available

            # Surrounding context frames
            prev_imgs = [keyframes[j] for j in range(max(0, i - CONTEXT_K), i)]
            next_imgs = [keyframes[j] for j in range(i + 1, min(len(keyframes), i + CONTEXT_K + 1))]
            transcript = _get_transcript_for_time(transcripts, pts)

            print(f"[Stage4] {vid}/{kf_name}  pts={pts:.1f}s  transcript={bool(transcript)}")
            caption = _caption_keyframe(kf_path, local_vlm, prev_imgs, next_imgs, transcript)

            rows.append({
                "video_id":      vid,
                "keyframe_name": kf_name,
                "frame_idx":     int(df_map.iloc[i]["frame_idx"]) if i < len(df_map) else 0,
                "pts_time":      pts,
                "shot_id":       shot_id,
                "caption":       caption,
            })

            # Save checkpoint every 50 captions
            if len(rows) % 50 == 0:
                pd.DataFrame(rows).to_parquet(CAPTIONS_PATH, index=False)
                print(f"  [Checkpoint] Saved {len(rows)} captions.")

            time.sleep(0.3)  # Gentle rate limit

    pd.DataFrame(rows).to_parquet(CAPTIONS_PATH, index=False)
    print(f"[Stage4] Done. {len(rows)} captions saved to {CAPTIONS_PATH}")
