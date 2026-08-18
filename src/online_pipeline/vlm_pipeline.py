"""Qwen2-VL inference wrapper for the Linux/CUDA submission runtime."""

from __future__ import annotations

import json
import re
from importlib.metadata import PackageNotFoundError, version

import torch
from PIL import Image


class VLMPipeline:
    def __init__(self, model_name: str = "Qwen/Qwen2-VL-7B-Instruct", device: str | None = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
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
        )
        self.processor = AutoProcessor.from_pretrained(self.model_name)
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
