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
        load_mode: str = "4bit",
    ):
        if load_mode not in {"4bit", "bf16"}:
            raise ValueError("load_mode must be '4bit' or 'bf16'")
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.local_files_only = local_files_only
        self.load_mode = load_mode
        self.model = None
        self.processor = None

    def load(self) -> None:
        if self.model is not None:
            return
        if self.device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Qwen2-VL inference requires a Linux/CUDA runtime (use Colab GPU).")

        if self.load_mode == "4bit":
            try:
                bnb_version = version("bitsandbytes")
            except PackageNotFoundError as exc:
                raise RuntimeError("Install bitsandbytes>=0.46.1 before loading Qwen2-VL.") from exc
            if tuple(int(part) for part in re.findall(r"\d+", bnb_version)[:3]) < (0, 46, 1):
                raise RuntimeError(f"bitsandbytes {bnb_version} is too old; install bitsandbytes>=0.46.1")
        elif not torch.cuda.is_bf16_supported():
            raise RuntimeError("The selected GPU does not support BF16; use --vlm-mode 4bit instead.")

        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration

        model_kwargs = {
            "device_map": "auto",
            "attn_implementation": "sdpa",
            "local_files_only": self.local_files_only,
        }
        if self.load_mode == "4bit":
            print(f"Loading VLM {self.model_name} in 4-bit CUDA mode...")
            model_kwargs.update({
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                ),
                "torch_dtype": torch.float16,
            })
        else:
            print(f"Loading VLM {self.model_name} in BF16 CUDA mode...")
            model_kwargs["torch_dtype"] = torch.bfloat16
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            **model_kwargs,
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

    def answer_question_details(self, image_path: str, question: str) -> dict[str, Any] | None:
        """Answer from visible evidence, returning machine-checkable confidence."""
        image = Image.open(image_path).convert("RGB")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": (
                    "Answer this visual question using only what is visible. Return ONLY valid JSON exactly as "
                    "{\"answer\": \"\", \"visible_evidence\": [], \"confidence\": 0}. "
                    "confidence is an integer from 0 to 3. Use 0 when the requested fact is not visible; do not guess. "
                    f"Question: {question}"
                )},
            ],
        }]
        parsed = self._json_object(self._generate_text(self._prepare(messages, image), max_new_tokens=96))
        if parsed is None or not isinstance(parsed.get("answer"), str) or isinstance(parsed.get("confidence"), bool):
            return None
        confidence = int(parsed["confidence"])
        if not 0 <= confidence <= 3:
            return None
        return {
            "answer": parsed["answer"].strip(),
            "visible_evidence": self._string_list(parsed.get("visible_evidence")),
            "confidence": confidence,
        }

    def answer_question(self, image_path: str, question: str) -> str:
        """Compatibility wrapper used by older UI paths."""
        details = self.answer_question_details(image_path, question)
        return str(details.get("answer", "")) if details else ""

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
            "You are preparing a visual known-item video search. Return ONLY valid JSON with keys "
            "visual_queries, metadata_queries, ocr_queries, visible_constraints, factual_entities. Every value is "
            "an array of concise English strings, even when the description is Vietnamese. The first visual_queries "
            "entry must be one complete, discriminative retrieval sentence; later entries may isolate distinct "
            "scenes/actions. metadata_queries "
            "contain titles, entities, places or event names useful in official video metadata. ocr_queries contain "
            "text likely to appear on screen. visible_constraints contain only discriminative things a frame can show. "
            "factual_entities contains named people, places, organisations or vehicles supported by the description "
            "or supplied offline evidence. Do not invent an event and do not omit original constraints.\n"
            f"Description: {english_query}\n"
            f"Offline evidence (possibly empty): {evidence_text[:6000]}"
        )
        raw = self._text_only_generate(prompt, max_new_tokens=256)
        parsed = self._json_object(raw)
        if parsed is None:
            raise ValueError(f"KIS query analysis is not valid JSON: {raw[:200]}")
        return {
            "visual_queries": self._string_list(parsed.get("visual_queries")),
            "metadata_queries": self._string_list(parsed.get("metadata_queries")),
            "ocr_queries": self._string_list(parsed.get("ocr_queries")),
            "visible_constraints": self._string_list(parsed.get("visible_constraints")),
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

    def read_text_in_image(self, image_path: str, question: str, lang: str = "vi") -> str:
        """OCR-mode: ask VLM to read and transcribe visible text relevant to the question."""
        from PIL import Image
        lang_out = "Vietnamese" if lang == "vi" else "English"
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": (
                    f"Question: {question}\n"
                    "First, accurately transcribe ALL text visible in this image, paying attention to banners, signs, and posters. "
                    f"Then, use the transcribed text to answer the question concisely in {lang_out}. "
                    "If no relevant text is visible, output ONLY: not visible."
                )},
            ],
        }]
        try:
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
            return self._generate_text(self._prepare(messages, image), max_new_tokens=128)
        except Exception as exc:
            print(f"[VLM] OCR error for {image_path}: {exc}")
            return "not visible"

    def analyze_vqa_query(self, english_question: str) -> dict:
        prompt = (
            "Analyze the question and output ONLY valid JSON with keys "
            "retrieval_description, vlm_question, paraphrases, metadata_queries, ocr_queries, objects_required, "
            "needs_fact_lookup, fact_query. metadata_queries and ocr_queries are optional retrieval text; "
            "do not invent facts.\n"
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

    def analyze_trake_query(self, description: str, events: list[str]) -> dict[str, Any]:
        """Create independent event queries and only explicit temporal edges.

        Edge indices are zero-based and are intentionally absent when the
        source wording does not assert an order. This prevents an invented
        chronological constraint from excluding a valid sequence.
        """
        prompt = (
            "Return ONLY valid JSON with keys retrieval_queries, event_queries, temporal_edges. "
            "retrieval_queries is an array for selecting the target video. event_queries is an array with exactly "
            f"{len(events)} arrays, one per listed event, containing visual paraphrases. temporal_edges is an array "
            "of [before_index, after_index] zero-based pairs ONLY where the description explicitly states before, "
            "after, then, next, followed by, or another unambiguous temporal relation. Do not assume the numbered "
            "event list is chronological.\n"
            f"Description: {description}\nEvents: {json.dumps(events, ensure_ascii=False)}"
        )
        parsed = self._json_object(self._text_only_generate(prompt, max_new_tokens=384))
        if parsed is None:
            raise ValueError("TRAKE query analysis is not valid JSON")
        raw_events = parsed.get("event_queries")
        event_queries: list[list[str]] = []
        if isinstance(raw_events, list):
            for index in range(len(events)):
                value = raw_events[index] if index < len(raw_events) else []
                event_queries.append(self._string_list(value))
        else:
            event_queries = [[] for _ in events]
        edges: list[tuple[int, int]] = []
        for edge in parsed.get("temporal_edges", []):
            if not isinstance(edge, list | tuple) or len(edge) != 2:
                continue
            try:
                before, after = int(edge[0]), int(edge[1])
            except (TypeError, ValueError):
                continue
            if 0 <= before < len(events) and 0 <= after < len(events) and before != after and (before, after) not in edges:
                edges.append((before, after))
        return {
            "retrieval_queries": self._string_list(parsed.get("retrieval_queries")),
            "event_queries": event_queries,
            "temporal_edges": edges,
        }

    def score_event_match_details(self, image_path: str, event: str) -> dict[str, Any] | None:
        """Return one rankable visual-evidence vote for a TRAKE event."""
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": (
                    "Does this frame show the event below? Return ONLY valid JSON exactly as "
                    "{\"score\": 0, \"visible_evidence\": []}. score is 0..3; use 0 if the event is not visible. "
                    f"Event: {event}"
                )},
            ],
        }]
        try:
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
            parsed = self._json_object(self._generate_text(self._prepare(messages, image), max_new_tokens=48))
            if parsed is None or isinstance(parsed.get("score"), bool):
                return None
            score = int(parsed["score"])
            if not 0 <= score <= 3:
                return None
            return {"score": score, "visible_evidence": self._string_list(parsed.get("visible_evidence"))}
        except Exception as exc:
            print(f"[TRAKE] Qwen event scoring skipped for {image_path}: {exc}")
            return None

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
