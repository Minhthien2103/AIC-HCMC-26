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
        if not torch.cuda.is_available():
            raise RuntimeError("Qwen2-VL 4-bit inference requires a CUDA-capable GPU.")
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
            max_memory={0: "12GB", "cpu": "24GB"},
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

    def answer_question(self, image_path: str, question: str, lang: str = "vi") -> str:
        image = Image.open(image_path).convert("RGB")
        lang_instruction = "Answer in Vietnamese." if lang == "vi" else "Answer in English."
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": (
                    f"Question: {question}\n\n"
                    f"{lang_instruction} "
                    "Answer concisely based ONLY on what is clearly visible in the image. "
                    "Make sure to answer ALL parts of the question (e.g. if asked about an object AND glasses, answer BOTH). "
                    "Be specific: name the exact object, color, brand if readable. "
                    "If something is not visible or unclear, say 'not visible'. "
                    "Do not guess. Max 80 characters."
                )},
            ],
        }]
        return self._generate_text(self._prepare(messages, image), max_new_tokens=64)
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
            "You are a video retrieval assistant. Analyze the question and output ONLY valid JSON with these keys:\n"
            "  retrieval_description: Scene description for CLIP image search. ONLY include KNOWN FACTS from the question. OMIT any unknown details the user is asking about (e.g. if asked 'is he wearing glasses?', DO NOT put glasses here). OMIT clothing/background colors. Focus on core ACTIONS and OBJECTS.\n"
            "  subject_description: Short description of the PERSON only (role, clothing color, accessories) for verification.\n"
            "  vlm_question: A direct factual question to ask the VLM. MUST include ALL questions/unknowns from the user. Write in the SAME LANGUAGE as the input question. (e.g. if input is 'đang cầm gì và có đeo kính không?', this must be 'Người đó đang cầm gì và có đeo kính không?').\n"
            "  paraphrases: 2 alternative retrieval_description strings using different wording.\n"
            "  objects_required: List of physical objects that must appear in the frame.\n"
            "  needs_fact_lookup: false\n"
            "  fact_query: \"\"\n"
            "EXAMPLES:\n"
            "Q: 1 người dẫn truyền hình nam, mặc áo xanh, đang cầm gì và có đeo kính không?\n"
            "A: {\"retrieval_description\": \"A male TV presenter holding an object.\", \"subject_description\": \"Male TV presenter wearing a blue shirt.\", \"vlm_question\": \"Người đó đang cầm cái gì trên tay và có đeo kính không?\"}\n"
            "IMPORTANT: retrieval_description must NOT mention clothing colors or background colors. MUST omit unknowns.\n"
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
                f"Is there a person who appears to be a TV presenter, news anchor, or public speaker in this image? Answer ONLY yes or no."
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