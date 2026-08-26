"""
Stage 5 - Shot-Level Temporal Memory (ReCap)
Input:  data/captions.parquet   (video_id, keyframe_name, shot_id, caption)
Output: data/shot_summaries.parquet (video_id, shot_id, summary, memory_context)
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

CAPTIONS_PATH      = config.DATA_DIR / "captions.parquet"
SHOT_SUMMARIES_PATH = config.DATA_DIR / "shot_summaries.parquet"


def _summarize_shot(shot_captions: list[str], memory_context: str, local_vlm=None) -> tuple[str, str]:
    """Returns (shot_summary, updated_memory_context)."""
    captions_text = "\n".join(f"- {c}" for c in shot_captions)

    prompt = (
        "You are building a temporal memory of a video by processing it shot by shot.\n\n"
        f"Accumulated memory from previous shots:\n{memory_context or '(This is the first shot)'}\n\n"
        f"Captions of keyframes in the CURRENT shot:\n{captions_text}\n\n"
        "Do two things:\n"
        "1. Write a concise summary (1-2 sentences) of what happens in the CURRENT shot, "
        "using the accumulated memory for context.\n"
        "2. Write an updated cumulative memory (max 3 sentences) covering everything that "
        "has happened in the video so far.\n\n"
        "Respond in this exact JSON format:\n"
        '{"summary": "...", "memory": "..."}'
    )
    try:
        if local_vlm:
            raw = local_vlm._text_only_generate(prompt, max_new_tokens=256).strip()
        else:
            response = MODEL.generate_content(prompt)
            raw = response.text.strip()
        
        import json, re
        raw = re.sub(r"`json|`", "", raw).strip()
        data = json.loads(raw)
        return data.get("summary", raw), data.get("memory", memory_context)
    except Exception as e:
        print(f"  [Stage5 ERROR] {e}")
        return " ".join(shot_captions[:2]), memory_context


def run_stage5(video_ids: list[str] | None = None, resume: bool = True, use_local: bool = False):
    if use_local:
        from src.online_pipeline.vlm_pipeline import VLMPipeline
        local_vlm = VLMPipeline()
        local_vlm.load()
    else:
        local_vlm = None

    if not CAPTIONS_PATH.exists():
        print("[Stage5] captions.parquet not found. Run Stage 4 first.")
        return

    df_cap = pd.read_parquet(CAPTIONS_PATH)
    existing = set()
    rows = []

    if resume and SHOT_SUMMARIES_PATH.exists():
        df_exist = pd.read_parquet(SHOT_SUMMARIES_PATH)
        rows = df_exist.to_dict("records")
        existing = set(zip(df_exist.video_id, df_exist.shot_id))
        print(f"[Stage5] Resuming: {len(existing)} shots already summarized.")

    if video_ids is None:
        video_ids = sorted(df_cap.video_id.unique())

    for vid in video_ids:
        df_vid = df_cap[df_cap.video_id == vid].sort_values("keyframe_name")
        shots = df_vid.groupby("shot_id")

        memory_context = ""
        for shot_id, df_shot in sorted(shots, key=lambda x: x[0]):
            if (vid, shot_id) in existing:
                existing_shot = next((r for r in rows if r["video_id"] == vid and r["shot_id"] == shot_id), None)
                if existing_shot:
                    memory_context = existing_shot.get("memory_context", "")
                continue

            captions = df_shot["caption"].dropna().tolist()
            if not captions:
                continue

            print(f"[Stage5] {vid} shot={shot_id}  ({len(captions)} frames)")
            summary, memory_context = _summarize_shot(captions, memory_context, local_vlm)

            rows.append({
                "video_id":       vid,
                "shot_id":        shot_id,
                "summary":        summary,
                "memory_context": memory_context,
            })

            if len(rows) % 20 == 0:
                pd.DataFrame(rows).to_parquet(SHOT_SUMMARIES_PATH, index=False)
                print(f"  [Checkpoint] Saved {len(rows)} shot summaries.")

            time.sleep(0.3)

    pd.DataFrame(rows).to_parquet(SHOT_SUMMARIES_PATH, index=False)
    print(f"[Stage5] Done. {len(rows)} shot summaries saved.")
