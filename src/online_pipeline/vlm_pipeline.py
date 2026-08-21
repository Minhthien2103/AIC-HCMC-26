"""Qwen2-VL inference wrapper for the Linux/CUDA submission runtime."""

from __future__ import annotations

import json
import re
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import torch
from PIL import Image


class VLMPipeline:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        device: str | None = None,
        *,
        local_files_only: bool = False,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.local_files_only = local_files_only
        self.model = None
        self.processor = None

    def load(self) -> None:
        if self.model is not None:
            return
        if self.device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Qwen2-VL 4-bit inference requires a Linux/CUDA runtime (use Colab GPU).")

        try:
            bnb_version = version("bitsandbytes")
        except PackageNotFoundError as exc:
            raise RuntimeError("Install bitsandbytes>=0.46.1 before loading Qwen2-VL.") from exc
        if tuple(int(part) for part in re.findall(r"\d+", bnb_version)[:3]) < (0, 46, 1):
            raise RuntimeError(f"bitsandbytes {bnb_version} is too old; install bitsandbytes>=0.46.1")

        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration

        print(f"Loading VLM {self.model_name} in 4-bit CUDA mode...")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
            local_files_only=self.local_files_only,
        )
        self.processor = AutoProcessor.from_pretrained(self.model_name, local_files_only=self.local_files_only)
        self.model.eval()
        print("VLM loaded successfully!")

    def _input_device(self) -> torch.device:
        if self.model is None:
            return torch.device("cuda")
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cuda")

    def _prepare(self, messages: list[dict], image: Image.Image | None = None):
        if self.model is None or self.processor is None:
            self.load()
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        kwargs = {"text": [text], "padding": True, "return_tensors": "pt"}
        if image is not None:
            kwargs["images"] = [image]
        inputs = self.processor(**kwargs)
        return inputs.to(self._input_device())

    def _generate_text(self, inputs, max_new_tokens: int) -> str:
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
            )
        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        return self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    def answer_question(self, image_path: str, question: str) -> str:
        image = Image.open(image_path).convert("RGB")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": (
                    f"{question}\n\n"
                    "Answer using only clearly visible information. Return only the answer, with no Markdown, "
                    "no explanation, and no more than 100 characters. If the requested detail is not visible, "
                    "say that it is not visible. Do not guess."
                )},
            ],
        }]
        return self._generate_text(self._prepare(messages, image), max_new_tokens=48)

    def extract_events(self, video_desc: str) -> list[str]:
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": (
                "Extract a chronological list of discrete events. Output ONLY a comma-separated list in English. "
                f"Description: {video_desc}"
            )}],
        }]
        raw = self._generate_text(self._prepare(messages), max_new_tokens=128)
        return [event.strip() for event in raw.split(",") if event.strip()]

    def _text_only_generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        return self._generate_text(self._prepare(messages), max_new_tokens=max_new_tokens)

    @staticmethod
    def _json_object(raw: str) -> dict[str, Any] | None:
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if not match:
            return None
        try:
            value = json.loads(match.group(0))
        except (json.JSONDecodeError, TypeError):
            return None
        return value if isinstance(value, dict) else None

    @staticmethod
    def _string_list(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return list(dict.fromkeys(
            text for text in (str(item).strip() for item in value) if text
        ))

    def analyze_kis_query(self, english_query: str, *, evidence_text: str = "") -> dict[str, list[str]]:
        """Produce additive retrieval constraints without replacing the query.

        ``evidence_text`` is read from an offline cache prepared before the
        final run. It may clarify named entities, but never overrides what the
        user asked to see in the supplied query.
        """
        prompt = (
            "You are preparing a visual known-item video search. Return ONLY valid JSON with "
            "keys retrieval_queries, must_have, expansions, factual_entities. All values are arrays of concise "
            "English strings. "
            "retrieval_queries must restate visible scenes or actions from the description. must_have must contain "
            "only visible attributes that distinguish the target. expansions may contain a synonym or a fact "
            "supported by the supplied offline evidence, but never invent an event. factual_entities must contain "
            "named people, places, organisations or vehicles that could help retrieve official metadata. Do not "
            "omit the original constraints.\n"
            f"Description: {english_query}\n"
            f"Offline evidence (possibly empty): {evidence_text[:6000]}"
        )
        raw = self._text_only_generate(prompt, max_new_tokens=256)
        parsed = self._json_object(raw)
        if parsed is None:
            raise ValueError(f"KIS query analysis is not valid JSON: {raw[:200]}")
        return {
            "retrieval_queries": self._string_list(parsed.get("retrieval_queries")),
            "must_have": self._string_list(parsed.get("must_have")),
            "expansions": self._string_list(parsed.get("expansions")),
            "factual_entities": self._string_list(parsed.get("factual_entities")),
        }

    def score_kis_match_details(
        self,
        image_path: str,
        original_query: str,
        must_have: list[str],
    ) -> dict[str, Any] | None:
        """Return Qwen visual evidence used as one rank-fusion source."""
        criteria = "\n".join(f"- {item}" for item in must_have) or "- Use the original description."
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": (
                    "Score how well this image matches the visual known-item search below. "
                    "Return ONLY valid JSON exactly like {\"score\": 0, \"visible_requirements\": []}. "
                    "Use 0 for unrelated/contradicted, 1 for only broad context, 2 for most visible "
                    "requirements, and 3 for all visible requirements. Do not infer details that cannot "
                    "be seen in the image. visible_requirements must list only requirements directly visible in "
                    "this image.\n"
                    f"Original description: {original_query}\n"
                    f"Visible requirements:\n{criteria}"
                )},
            ],
        }]
        try:
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
            parsed = self._json_object(self._generate_text(self._prepare(messages, image), max_new_tokens=32))
            if parsed is None or isinstance(parsed.get("score"), bool):
                return None
            score = int(parsed["score"])
            if not 0 <= score <= 3:
                return None
            return {
                "score": score,
                "visible_requirements": self._string_list(parsed.get("visible_requirements")),
            }
        except Exception as exc:
            print(f"[KIS] Qwen visual re-rank skipped for {image_path}: {exc}")
            return None

    def score_kis_match(
        self,
        image_path: str,
        original_query: str,
        must_have: list[str],
    ) -> int | None:
        """Compatibility wrapper for existing callers."""
        details = self.score_kis_match_details(image_path, original_query, must_have)
        return int(details["score"]) if details is not None else None

    def analyze_vqa_query(self, english_question: str) -> dict:
        prompt = (
            "Analyze the question and output ONLY valid JSON with keys "
            "retrieval_description, vlm_question, paraphrases, objects_required, "
            "needs_fact_lookup, fact_query. Do not invent facts.\n"
            f"Question: {english_question}"
        )
        raw = self._text_only_generate(prompt, max_new_tokens=256)
        try:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
                return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, TypeError):
            pass
        print(f"[VLM] Could not parse query analysis: {raw[:200]}")
        return {
            "needs_fact_lookup": False,
            "fact_query": "",
            "retrieval_description": english_question,
            "vlm_question": english_question,
            "paraphrases": [],
            "objects_required": [],
        }

    def verify_frame(self, image_path: str, retrieval_description: str) -> bool:
        messages = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": (
                f"Does this image match the description: {retrieval_description}? Answer ONLY yes or no."
            )}],
        }]
        try:
            answer = self._generate_text(
                self._prepare(messages, Image.open(image_path).convert("RGB")),
                max_new_tokens=5,
            ).lower()
            return answer.startswith("yes")
        except Exception as exc:
            print(f"[VLM] verify_frame error: {exc}")
            return True
