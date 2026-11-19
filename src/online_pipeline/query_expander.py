import logging
from typing import List
import torch
from src.online_pipeline.query_translation import (translation_messages, parse_translation,
                                                    expansion_messages, parse_expansions)

logger = logging.getLogger(__name__)

class QueryExpander:
    """Translate first, then expand with a frozen primary query using a local SLM."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-4B-Instruct-2507",
        device: str = "cuda",
        load_in_4bit: bool = True
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        self.tokenizer = None
        self.model = None
        self.cache = {}
        self.load_in_4bit = load_in_4bit and (self.device == "cuda")

        self._load_model()

    def _load_model(self):
        try:
            logger.info(f"Loading local LLM for Query Expansion: {self.model_id} (4-bit={self.load_in_4bit})...")
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, local_files_only=True)
            except Exception:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, local_files_only=False)

            if self.load_in_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16
                )
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_id,
                        quantization_config=bnb_config,
                        device_map="cuda",
                        local_files_only=True
                    )
                except Exception:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_id,
                        quantization_config=bnb_config,
                        device_map="cuda",
                        local_files_only=False
                    )
            else:
                dtype = torch.bfloat16 if self.device == "cpu" else torch.float16
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    torch_dtype=dtype,
                    device_map=None if self.device == "cpu" else "auto",
                    local_files_only=True
                )
                if self.device == "cpu":
                    self.model.to("cpu")
            self.model.eval()
            logger.info("QueryExpander LLM loaded successfully.")
        except Exception as e:
            logger.warning(f"Failed to load QueryExpander LLM ({e}). Will use fallback translation.")
            self.model = None

    def _generate(self, messages, max_new_tokens=192):
        chat_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = self.tokenizer([chat_text], return_tensors="pt").to(self.device)
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generated = output[0][inputs.input_ids.shape[1]:]
        # A truncated JSON response must never become a primary query.
        if len(generated) >= max_new_tokens:
            raise ValueError("SLM response exceeded token budget")
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def expand(self, query: str) -> List[str]:
        """Return a contextual translation followed by up to two paraphrases."""
        clean_q = query.strip()
        if not clean_q:
            return []
        if clean_q in self.cache:
            return self.cache[clean_q]

        primary = None
        if self.model is not None and self.tokenizer is not None:
            try:
                primary = parse_translation(self._generate(translation_messages(clean_q)), clean_q)
            except Exception as exc:
                logger.warning("Primary translation failed: %s", exc)

        if primary is None:
            from src.utils.translation import translate_vi_to_en
            translated = translate_vi_to_en(clean_q)
            primary = translated.strip() if translated and translated.strip() else clean_q
            # Do not cache transient fallback errors or expand an unverified fallback.
            return [primary]

        results = [primary]
        try:
            results = parse_expansions(self._generate(expansion_messages(primary)), primary)
        except Exception as exc:
            logger.warning("Paraphrases failed; keeping primary translation: %s", exc)
        self.cache[clean_q] = results
        logger.info("[Query Expansion] %r -> %s", clean_q, results)
        return results
