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

load_dotenv()
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
from src import config

_MODEL = None
def get_gemini_model():
    global _MODEL
    if _MODEL is None:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        _MODEL = genai.GenerativeModel("gemini-3.1-pro")
    return _MODEL



CAPTIONS_PATH = config.DATA_DIR / "captions.parquet"
TRANSCRIPTS_PATH = config.DATA_DIR / "transcripts.parquet"
CONTEXT_K = 2  # Surrounding frames to include as context


def _load_transcripts() -> dict[str, list[dict]]:
    """Load transcripts indexed by video_id -> list of {start, end, text}."""
    path = getattr(config, "SUBTITLES_JSON_PATH", config.DATA_DIR / "extracted_subtitles" / "all_subtitles.json")
    if not path.exists():
        return {}
        
    try:
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        result = {}
        for item in data:
            vid = item.get("video_id")
            if not vid: continue
            
            # Filter out non-spoken closed captions (e.g. "[music playing]")
            text = str(item.get("subtitles", "")).strip()
            if text.startswith("[") and text.endswith("]"):
                continue
            
            result.setdefault(vid, []).append({
                "start": float(item.get("timestamp_start", 0)),
                "end":   float(item.get("timestamp_end", 0)),
                "text":  text,
            })
        return result
    except Exception as e:
        print(f"[Stage4] Failed to load subtitles: {e}")
        return {}


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
            response = get_gemini_model().generate_content(parts)
            return response.text.strip()
    except Exception as e:
        print(f"  [Caption ERROR] {e}")
        return ""



def _detect_shots(video_id: str, df_map: pd.DataFrame) -> dict[int, int]:
    """
    Detects shots using jpeg_size.npy if available.
    Returns a dict mapping frame_idx -> shot_id.
    """
    shot_map = {}
    default_shot = 0
    
    jpeg_dir = getattr(config, "JPEG_SIZE_DIR", config.DATA_DIR / "jpeg_sizes")
    npy_path = jpeg_dir / f"{video_id}.npy"
    
    if not npy_path.exists():
        return {int(row["frame_idx"]): default_shot for _, row in df_map.iterrows()}
        
    try:
        import numpy as np
        sizes = np.load(npy_path)
        
        if len(sizes) > 1:
            diffs = np.abs(np.diff(sizes))
            threshold = max(np.median(diffs) * 3, 50000) 
            
            shot_id = 0
            shot_map[int(df_map.iloc[0]["frame_idx"])] = shot_id
            
            for i in range(1, len(sizes)):
                if i - 1 < len(diffs) and diffs[i-1] > threshold:
                    shot_id += 1
                if i < len(df_map):
                    shot_map[int(df_map.iloc[i]["frame_idx"])] = shot_id
        else:
            if len(df_map) > 0:
                shot_map[int(df_map.iloc[0]["frame_idx"])] = 0
                
        return shot_map
    except Exception as e:
        print(f"[Stage4] Error computing shots for {video_id}: {e}")
        return {int(row["frame_idx"]): default_shot for _, row in df_map.iterrows()}

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
        keyframes = sorted((config.KEYFRAMES_DIR / vid).glob("*.jpg"), key=lambda p: p.name)
        
        transcripts = all_transcripts.get(vid, [])
        shot_map = _detect_shots(vid, df_map)

        for i, kf_path in enumerate(keyframes):
            kf_name = kf_path.stem  # e.g. "001"
            if (vid, kf_name) in existing:
                continue

            # Get frame metadata
            idx = i
            pts = float(df_map.iloc[i]["pts_time"]) if i < len(df_map) else 0.0
            frame_idx = int(df_map.iloc[i]["frame_idx"]) if i < len(df_map) else 0
            shot_id = shot_map.get(frame_idx, 0)


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
