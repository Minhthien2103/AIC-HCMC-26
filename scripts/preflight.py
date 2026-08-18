#!/usr/bin/env python3
"""Check the CUDA/Qwen2-VL runtime before a batch submission run."""

from __future__ import annotations

import argparse
import sys
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--skip-model", action="store_true")
    args = parser.parse_args()

    import torch

    print(f"torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("PREFLIGHT FAILED: enable a CUDA GPU runtime (Colab Runtime > Change runtime type > T4 GPU).")
        return 1
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    try:
        import bitsandbytes  # noqa: F401

        print(f"bitsandbytes: {version('bitsandbytes')}")
        if tuple(int(part) for part in __import__('re').findall(r"\d+", version('bitsandbytes'))[:3]) < (0, 46, 1):
            raise RuntimeError("bitsandbytes>=0.46.1 is required")
    except Exception as exc:
        print(f"PREFLIGHT FAILED: bitsandbytes check: {exc}")
        return 1

    if not args.skip_model:
        try:
            from src.online_pipeline.vlm_pipeline import VLMPipeline

            vlm = VLMPipeline(model_name=args.model, device="cuda")
            vlm.load()
            print("Qwen2-VL model load: OK")
        except Exception as exc:
            print(f"PREFLIGHT FAILED: Qwen2-VL load: {exc}")
            return 1
    print("PREFLIGHT PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
