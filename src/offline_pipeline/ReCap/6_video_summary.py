"""
Stage 6 - Video-Level Summary
Input:  data/shot_summaries.parquet  (video_id, shot_id, summary)
Output: data/video_summaries.parquet (video_id, summary)
"""
import os, sys, time
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
from src import config

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = genai.GenerativeModel("gemini-3.1-pro")

SHOT_SUMMARIES_PATH  = config.DATA_DIR / "shot_summaries.parquet"
VIDEO_SUMMARIES_PATH = config.DATA_DIR / "video_summaries.parquet"


def _summarize_video(shot_summaries: list[str], local_vlm=None) -> str:
    ordered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(shot_summaries))
    prompt = (
        "Below are chronological shot-level summaries of a video.\n\n"
        f"{ordered}\n\n"
        "Write ONE concise paragraph (3-5 sentences) describing the overall content "
        "of this video, covering the main topics, subjects, locations, and sequence of events. "
        "Output ONLY the summary paragraph."
    )
    try:
        if local_vlm:
            return local_vlm._text_only_generate(prompt, max_new_tokens=256).strip()
        else:
            response = MODEL.generate_content(prompt)
            return response.text.strip()
    except Exception as e:
        print(f"  [Stage6 ERROR] {e}")
        return " ".join(shot_summaries[:3])


def run_stage6(video_ids: list[str] | None = None, resume: bool = True, use_local: bool = False):
    if use_local:
        from src.online_pipeline.vlm_pipeline import VLMPipeline
        local_vlm = VLMPipeline()
        local_vlm.load()
    else:
        local_vlm = None

    if not SHOT_SUMMARIES_PATH.exists():
        print("[Stage6] shot_summaries.parquet not found. Run Stage 5 first.")
        return

    df_shots = pd.read_parquet(SHOT_SUMMARIES_PATH)
    existing = set()
    rows = []

    if resume and VIDEO_SUMMARIES_PATH.exists():
        df_exist = pd.read_parquet(VIDEO_SUMMARIES_PATH)
        rows = df_exist.to_dict("records")
        existing = set(df_exist.video_id)
        print(f"[Stage6] Resuming: {len(existing)} videos already summarized.")

    if video_ids is None:
        video_ids = sorted(df_shots.video_id.unique())

    for vid in video_ids:
        if vid in existing:
            continue

        df_vid = df_shots[df_shots.video_id == vid].sort_values("shot_id")
        summaries = df_vid["summary"].dropna().tolist()
        if not summaries:
            continue

        print(f"[Stage6] {vid}  ({len(summaries)} shots)")
        video_summary = _summarize_video(summaries, local_vlm)

        rows.append({"video_id": vid, "summary": video_summary})

        if len(rows) % 10 == 0:
            pd.DataFrame(rows).to_parquet(VIDEO_SUMMARIES_PATH, index=False)
            print(f"  [Checkpoint] Saved {len(rows)} video summaries.")

        time.sleep(0.3)

    pd.DataFrame(rows).to_parquet(VIDEO_SUMMARIES_PATH, index=False)
    print(f"[Stage6] Done. {len(rows)} video summaries saved.")
