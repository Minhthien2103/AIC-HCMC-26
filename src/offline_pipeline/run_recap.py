"""
run_recap.py — Master orchestrator for the full ReCap pipeline.

Usage:
    # Run all 4 stages on all videos
    python scripts/run_recap.py

    # Run specific stages only
    python scripts/run_recap.py --stages 4 5

    # Run on specific videos only
    python scripts/run_recap.py --videos L21_V001 L21_V002

    # Start fresh (ignore checkpoints)
    python scripts/run_recap.py --no-resume
"""
import sys, argparse
from importlib import import_module
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))


def _stage_runner(stage: int):
    """Load a ReCap stage whose source filename starts with its stage number."""
    module_names = {
        4: "src.offline_pipeline.ReCap.4_keyframe_caption",
        5: "src.offline_pipeline.ReCap.5_shot_caption",
        6: "src.offline_pipeline.ReCap.6_video_summary",
        7: "src.offline_pipeline.ReCap.7_embed",
    }
    return import_module(module_names[stage])

def main():
    parser = argparse.ArgumentParser(description="Run the ReCap offline pipeline.")
    parser.add_argument("--stages", nargs="*", type=int, default=[4, 5, 6, 7],
                        help="Which stages to run (default: 4 5 6 7)")
    parser.add_argument("--videos", nargs="*", default=None,
                        help="Specific video IDs to process (default: all)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start fresh, ignore existing checkpoints")
    parser.add_argument("--local", action="store_true", help="Use local VLM (Qwen2-VL) instead of Gemini API")
    args = parser.parse_args()

    resume = not args.no_resume
    stages = set(args.stages)
    use_local = args.local

    print("=" * 60)
    print(f"ReCap Pipeline | Stages: {sorted(stages)} | Resume: {resume} | Local: {use_local}")
    if args.videos:
        print(f"Videos: {args.videos}")
    print("=" * 60)

    if 4 in stages:
        print("\n── Stage 4: Keyframe Captioning ──")
        _stage_runner(4).run_stage4(video_ids=args.videos, resume=resume, use_local=use_local)

    if 5 in stages:
        print("\n── Stage 5: Shot-Level Temporal Memory ──")
        _stage_runner(5).run_stage5(video_ids=args.videos, resume=resume, use_local=use_local)

    if 6 in stages:
        print("\n── Stage 6: Video-Level Summary ──")
        _stage_runner(6).run_stage6(video_ids=args.videos, resume=resume, use_local=use_local)

    if 7 in stages:
        print("\n── Stage 7: Embed + Ingest to Milvus & ElasticSearch ──")
        _stage_runner(7).run_stage7()

    print("\n✓ ReCap pipeline complete.")

if __name__ == "__main__":
    main()
