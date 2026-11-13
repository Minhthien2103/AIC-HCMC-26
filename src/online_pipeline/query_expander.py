import logging
import re
from typing import List
import torch

logger = logging.getLogger(__name__)

class QueryExpander:
    """Multi-query expansion and typo correction using local Qwen2.5-3B-Instruct."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-3B-Instruct",
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

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, local_files_only=True)

            if self.load_in_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    quantization_config=bnb_config,
                    device_map="cuda",
                    local_files_only=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    local_files_only=True
                )
            self.model.eval()
            logger.info("QueryExpander LLM loaded successfully.")
        except Exception as e:
            logger.warning(f"Failed to load QueryExpander LLM ({e}). Will use fallback translation.")
            self.model = None

    def expand(self, query: str) -> List[str]:
        """Expand a raw Vietnamese or English query into 3 distinct visual search prompts."""
        if not query or not query.strip():
            return []

        clean_q = query.strip()
        if clean_q in self.cache:
            return self.cache[clean_q]

        if self.model is None or self.tokenizer is None:
            # Fallback to direct translation
            from src.utils.translation import translate_vi_to_en
            trans = translate_vi_to_en(clean_q)
            return [trans] if trans else [clean_q]

        prompt = f"""You are an expert video retrieval query optimizer for text-to-video search.
Task:
1. Carefully detect and correct any typos, wrong tone marks, or phonetic misspellings in the user query (for example: "tay đưa xe đạp" -> "tay đua xe đạp" / cyclist, "ngừoi" -> "người", "xe dap" -> "xe đạp").
2. Translate and generate exactly 3 distinct English visual search descriptions tailored for SigLIP/CLIP keyframe retrieval:
   - Line 1: Detailed visual scene description (who, what they are doing, visual setting, clothing color)
   - Line 2: Subject and core action focused
   - Line 3: Key objects, colors, and visual tags comma-separated

Output format: Output ONLY the 3 English lines, one per line. No numbering, no prefixes, no explanation, no Vietnamese.

User Query: "{clean_q}"
English Visual Descriptions:"""

        try:
            messages = [{"role": "user", "content": prompt}]
            chat_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer([chat_text], return_tensors="pt").to(self.device)

            with torch.inference_mode():
                out = self.model.generate(**inputs, max_new_tokens=80, do_sample=False)

            raw_resp = self.tokenizer.batch_decode([out[0][len(inputs.input_ids[0]):]], skip_special_tokens=True)[0]
            lines = []
            for line in raw_resp.strip().split("\n"):
                cleaned = line.strip()
                cleaned = re.sub(r'^(Line\s*\d+:?|\d+[\.\)]|-|\*)\s*', '', cleaned, flags=re.IGNORECASE).strip()
                if cleaned and len(cleaned) > 3:
                    lines.append(cleaned)

            if lines:
                results = lines[:3]
                self.cache[clean_q] = results
                logger.info(f"[Query Expansion] '{clean_q}' -> {results}")
                return results
        except Exception as e:
            logger.warning(f"Error during LLM expansion for '{clean_q}': {e}")

        # Fallback if generation failed
        from src.utils.translation import translate_vi_to_en
        fallback = translate_vi_to_en(clean_q)
        res = [fallback] if fallback else [clean_q]
        self.cache[clean_q] = res
        return res
