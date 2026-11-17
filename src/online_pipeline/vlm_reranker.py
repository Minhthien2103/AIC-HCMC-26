"""VLM Reranker using Qwen2-VL-7B-Instruct (4-bit quantization).

Performs fine-grained cross-modal verification on Top-K visual candidates (e.g. Top 20-30).
Combines deep visual reasoning with initial SigLIP/OCR retrieval scores.
"""

from __future__ import annotations

import io
import logging
import re
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from src.online_pipeline.score_fusion import unit_score
from src.online_pipeline.diverse_candidates import select_diverse_indices

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def fetch_frame_image(video_id: str, frame_name: str, media_base: str = "http://localhost:8007") -> Optional[Image.Image]:
    """Load keyframe image from local filesystem or stream from media server."""
    # 1. Try local filesystem
    local_candidates = [
        REPO_ROOT / "data" / "keyframes" / str(video_id) / f"{str(frame_name).removesuffix('.jpg')}.jpg",
        REPO_ROOT / "data" / "keyframes" / f"{str(video_id)}_{str(frame_name).removesuffix('.jpg')}.jpg",
    ]
    for p in local_candidates:
        if p.is_file():
            try:
                return Image.open(p).convert("RGB")
            except Exception:
                pass

    # 2. Try streaming from media server
    url = f"{media_base}/keyframes/{video_id}/{str(frame_name).removesuffix('.jpg')}.jpg"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AIC-VLM-Reranker/1.0"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            return Image.open(io.BytesIO(resp.read())).convert("RGB")
    except Exception:
        return None


class VLMReranker:
    """Zero-shot Vision-Language Reranker using local Qwen2-VL-7B (4-bit)."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2-VL-7B-Instruct",
        device: str = "cuda",
        media_base: str = "http://localhost:8007",
        load_mode: str = "4bit"
    ):
        self.model_id = model_id
        self.device = device if torch.cuda.is_available() else "cpu"
        self.media_base = media_base
        self.load_mode = load_mode
        self.model = None
        self.processor = None
        self._is_loading = False

    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self):
        """Lazy load Qwen2-VL in 4-bit quantization on GPU."""
        if self.model is not None or self._is_loading:
            return

        self._is_loading = True
        try:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info(f"Loading single VLM model: {self.model_id} (4-bit={self.load_mode == '4bit'})...")
            from transformers import AutoProcessor, BitsAndBytesConfig
            try:
                from transformers import AutoModelForImageTextToText as AutoModelVLM
            except ImportError:
                try:
                    from transformers import AutoModelForVision2Seq as AutoModelVLM
                except ImportError:
                    from transformers import Qwen2VLForConditionalGeneration as AutoModelVLM

            model_kwargs = {
                "device_map": "auto" if self.device == "cuda" else None,
            }

            if self.load_mode == "4bit" and self.device == "cuda":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    llm_int8_enable_fp32_cpu_offload=True
                )
                model_kwargs["quantization_config"] = bnb_config
            else:
                model_kwargs["torch_dtype"] = torch.float16 if self.device == "cuda" else torch.float32

            # Load ONLY the specified model (no multiple models / fallbacks)
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model = AutoModelVLM.from_pretrained(self.model_id, **model_kwargs)
            self.model.eval()
            logger.info(f"VLM Reranker ({self.model_id}) loaded successfully into GPU!")
        except Exception as e:
            logger.error(f"Failed to load VLM Reranker: {e}")
            self.model = None
            self.processor = None
        finally:
            self._is_loading = False

    def score_single_frame(self, image: Image.Image, query: str) -> Optional[float]:
        """Evaluate how well an image matches the query on a scale of 0.0 to 1.0."""
        if self.model is None or self.processor is None:
            self.load()
        if self.model is None:
            return None

        # Resize image for fast inference (max 448x448)
        img_resized = image.copy()
        img_resized.thumbnail((448, 448))

        prompt = (
            f"You are a strict video search evaluator.\n"
            f"User Query: \"{query}\"\n"
            f"Task: Rate how well this image depicts the user query on a scale from 0 to 10 "
            f"(10 = perfect match, 5 = partially relevant, 0 = completely unrelated).\n"
            f"Output ONLY an integer number from 0 to 10. No explanation.\n"
            f"Score:"
        )

        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }]

        try:
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[img_resized], padding=True, return_tensors="pt")
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.inference_mode():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=4,
                    do_sample=False,
                    num_beams=1
                )

            gen_text = self.processor.batch_decode(
                out[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )[0].strip()

            # Parse integer 0-10
            match = re.search(r'\b(10|[0-9])\b', gen_text)
            if match:
                score_int = int(match.group(1))
                return float(score_int) / 10.0
            return None
        except Exception as e:
            logger.warning(f"Error during VLM scoring: {e}")
            return None

    def rerank_candidates(
        self,
        candidates: List[Dict[str, Any]],
        query: str,
        top_n: int = 20,
        vlm_weight: float = 0.65,
        diversity_index=None,
        diversity_pool: int = 100,
        diversity_similarity: float = 0.97,
        diversity_seconds: float = 3.0,
    ) -> List[Dict[str, Any]]:
        """Rerank up to top_n diverse candidates; retain all unselected frames."""
        if not candidates or not query or top_n <= 0:
            return [dict(c, vlm_selected=False) for c in candidates]

        if self.model is None:
            self.load()
        if self.model is None:
            return [dict(c, vlm_selected=False) for c in candidates]

        t0 = time.time()
        vlm_weight = unit_score(vlm_weight)
        selected = select_diverse_indices(candidates, diversity_index, top_n,
                                          diversity_pool, diversity_similarity, diversity_seconds)
        selected_set = set(selected)
        to_rerank = [dict(candidates[i], entity=dict(candidates[i].get("entity") or {}),
                          vlm_selected=True) for i in selected]
        remaining = [dict(c, vlm_selected=False) for i, c in enumerate(candidates) if i not in selected_set]

        logger.info(f"[VLM Rerank] Scoring Top {len(to_rerank)} frames for query: '{query[:50]}'...")

        for idx, c in enumerate(to_rerank):
            entity = c.get("entity", {}) if isinstance(c.get("entity"), dict) else {}
            v_id = entity.get("video") or c.get("video") or c.get("video_id", "")
            f_name = entity.get("frame_id") or c.get("frame_id") or c.get("keyframe_name", "")
            orig_score = unit_score(c.get("orig_score", c.get("score", c.get("distance", 0.0))))

            img = fetch_frame_image(v_id, f_name, media_base=self.media_base)
            vlm_score = self.score_single_frame(img, query) if img is not None else None
            if vlm_score is not None:
                vlm_score = unit_score(vlm_score)
                blended_score = (vlm_weight * vlm_score) + ((1.0 - vlm_weight) * orig_score)
                c["vlm_score"] = round(vlm_score, 3)
                c["orig_score"] = orig_score
                c["score"] = round(blended_score, 4)
                c["distance"] = round(blended_score, 4)
                c["vlm_verified"] = True
                if entity:
                    entity["vlm_score"] = round(vlm_score, 3)
                    entity["vlm_verified"] = True
            else:
                c["score"] = orig_score
                c["distance"] = orig_score
                c.pop("vlm_score", None)
                c["vlm_verified"] = False
                entity.pop("vlm_score", None)
                entity["vlm_verified"] = False

        # Sort reranked top candidates descending by new blended score
        to_rerank.sort(key=lambda x: x.get("score", 0.0), reverse=True)

        elapsed = time.time() - t0
        logger.info(f"[VLM Rerank] Finished reranking {len(to_rerank)} frames in {elapsed:.2f}s ({len(to_rerank)/elapsed:.1f} FPS)")

        return to_rerank + remaining
