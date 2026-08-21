#!/usr/bin/env python3
"""Check the CUDA/Qwen2-VL runtime before a batch submission run."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--kis-profile", choices=("fast", "full"), default="fast")
    parser.add_argument("--require-kis-assets", action="store_true")
    parser.add_argument("--vlm-mode", choices=("4bit", "bf16"), default="4bit")
    parser.add_argument("--min-free-gb", type=float, default=8.0)
    parser.add_argument("--offline", action="store_true", help="Require all models to be present in Hugging Face cache.")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    if not (repo_root / "src" / "config.py").exists():
        print(f"PREFLIGHT FAILED: invalid --repo-root {repo_root}")
        return 1
    sys.path.insert(0, str(repo_root))
    from src import config

    config.BASE_DIR = repo_root
    config.DATA_DIR = repo_root / "data"
    config.INDEX_DIR = repo_root / "indexes"
    config.FAISS_INDEX_PATH = config.INDEX_DIR / "faiss_clip.index"
    config.METADATA_PATH = config.INDEX_DIR / "metadata.parquet"
    config.MEDIA_TEXT_INDEX_PATH = config.INDEX_DIR / "media_e5.index"
    config.MEDIA_TEXT_RECORDS_PATH = config.INDEX_DIR / "media_e5_records.json"
    config.VITH_INDEX_PATH = config.INDEX_DIR / "faiss_vith.index"
    config.KEYFRAMES_DIR = config.DATA_DIR / "keyframes"

    try:
        with tempfile.NamedTemporaryFile(prefix="aic2026-preflight-", delete=True) as stream:
            stream.write(b"ok")
            stream.flush()
        print(f"temporary directory: OK ({tempfile.gettempdir()})")
    except Exception as exc:
        print(f"PREFLIGHT FAILED: no writable temporary directory: {exc}")
        return 1
    free = shutil.disk_usage(tempfile.gettempdir()).free / (1024 ** 3)
    print(f"temporary disk free: {free:.1f} GB")
    if free < args.min_free_gb:
        print(f"PREFLIGHT FAILED: temporary disk has {free:.1f} GB free; need at least {args.min_free_gb:.1f} GB. Clear /content before model loading.")
        return 1
    required = [config.FAISS_INDEX_PATH, config.METADATA_PATH, config.KEYFRAMES_DIR]
    if args.kis_profile == "full":
        required.append(config.VITH_INDEX_PATH)
    missing = [path for path in required if not path.exists()]
    if missing:
        print("PREFLIGHT FAILED: missing profile assets: " + ", ".join(str(path) for path in missing))
        return 1
    optional_fast = [config.MEDIA_TEXT_INDEX_PATH, config.MEDIA_TEXT_RECORDS_PATH]
    missing_fast = [path for path in optional_fast if not path.exists()]
    if missing_fast and args.require_kis_assets:
        print("PREFLIGHT FAILED: missing fast KIS assets: " + ", ".join(str(path) for path in missing_fast))
        return 1
    if missing_fast:
        print("KIS profile warning: media-E5 assets missing; run will use ViT-B/Qwen fallback")
    else:
        print("KIS fast assets: ViT-B + media-E5 OK")
    cache_roots = [
        Path(value) for value in (
            os.environ.get("HF_HUB_CACHE"),
            str(Path(os.environ["HF_HOME"]) / "hub") if os.environ.get("HF_HOME") else None,
            str(Path.home() / ".cache" / "huggingface" / "hub"),
        ) if value
    ]
    model_dirs = [
        "models--Qwen--Qwen2-VL-7B-Instruct",
        "models--facebook--mbart-large-50-many-to-many-mmt",
    ]
    if not missing_fast:
        model_dirs.append("models--intfloat--multilingual-e5-base")
    missing_models = [
        model for model in model_dirs
        if not any((root / model / "snapshots").exists() and any((root / model / "snapshots").iterdir()) for root in cache_roots)
    ]
    if missing_models:
        message = "model cache missing: " + ", ".join(missing_models)
        if args.offline:
            print(f"PREFLIGHT FAILED: {message}")
            return 1
        print(f"PREFLIGHT WARNING: {message}; first online preparation may download them")
    else:
        print("model cache: Qwen/E5/mBART OK")
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import torch

    print(f"torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("PREFLIGHT FAILED: enable a CUDA GPU runtime (A100 recommended for BF16).")
        return 1
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    if args.vlm_mode == "4bit":
        try:
            import bitsandbytes  # noqa: F401

            print(f"bitsandbytes: {version('bitsandbytes')}")
            if tuple(int(part) for part in __import__('re').findall(r"\d+", version('bitsandbytes'))[:3]) < (0, 46, 1):
                raise RuntimeError("bitsandbytes>=0.46.1 is required")
        except Exception as exc:
            print(f"PREFLIGHT FAILED: bitsandbytes check: {exc}")
            return 1
    elif not torch.cuda.is_bf16_supported():
        print("PREFLIGHT FAILED: selected GPU does not support BF16; use --vlm-mode 4bit")
        return 1

    if not args.skip_model:
        try:
            from src.online_pipeline.vlm_pipeline import VLMPipeline

            vlm = VLMPipeline(
                model_name=args.model,
                device="cuda",
                local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1",
                load_mode=args.vlm_mode,
            )
            vlm.load()
            print("Qwen2-VL model load: OK")
        except Exception as exc:
            print(f"PREFLIGHT FAILED: Qwen2-VL load: {exc}")
            return 1
    print("PREFLIGHT PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
