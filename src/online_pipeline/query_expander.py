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
        self.load_error = None
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
                    self.model_id, quantization_config=bnb_config,
                    device_map="cuda", local_files_only=True
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
            self.load_error = str(e)
            logger.warning(f"Local QueryExpander unavailable ({e}). Vietnamese search will require a working local model; English remains available.")
            self.model = None

    def expand(self, query: str) -> List[str]:
        """Expand a raw Vietnamese or English query into 3 distinct visual search prompts."""
        if not query or not query.strip():
            return []

        clean_q = query.strip()
        if clean_q in self.cache:
            return self.cache[clean_q]

        if self.model is None or self.tokenizer is None:
            raise RuntimeError(self.load_error or "Local translation model is not loaded")

        prompt = f"""You are an expert video retrieval query optimizer for text-to-video search.
Task:
1. Carefully detect and correct any typos, wrong tone marks, or phonetic misspellings in the user query (for example: "tay đưa xe đạp" -> "tay đua xe đạp" / cyclist, "ngừoi" -> "người", "xe dap" -> "xe đạp").
2. Translate and generate exactly 3 distinct English visual search descriptions tailored for SigLIP/CLIP keyframe retrieval:
   - Line 1: Faithful, natural English translation preserving ALL explicit constraints, actions, relationships, negations, numbers and colors. Translate meaning in context, NEVER word by word. Vietnamese "tay đua xe đạp" means "cyclist" or "bicycle racer", NEVER "hand racing cycle"; "tay" here refers to a person, not a body part. If already English, preserve its meaning.
   - Line 2: Subject and core action focused
   - Line 3: Key objects, colors, and visual tags comma-separated
3. All three lines must use ONLY information in the input. Do not invent setting, clothing, objects, actions, or visible text. Line 1 is the authoritative query, not a creative expansion.

Output format: Output ONLY the 3 English lines, one per line. No numbering, no prefixes, no explanation, no Vietnamese.

User Query: "{clean_q}"
English Visual Descriptions:"""

        try:
            messages = [{"role": "user", "content": prompt}]
            chat_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer([chat_text], return_tensors="pt").to(self.device)

            with torch.inference_mode():
                out = self.model.generate(**inputs, max_new_tokens=192, do_sample=False)

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

        raise RuntimeError("Local query expansion did not produce usable output")
