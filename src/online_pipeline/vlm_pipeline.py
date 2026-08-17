import torch
from PIL import Image

class VLMPipeline:
    def __init__(self, model_name="Qwen/Qwen2-VL-7B-Instruct"):
        self.model_name = model_name
        self.model = None
        self.processor = None
        
    def load(self):
        if self.model is not None:
            return
            
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
        print(f"Loading VLM {self.model_name} in 4-bit with CPU offload...")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            # llm_int8_enable_fp32_cpu_offload removed due to Qwen2-VL vision tower meta tensor bug
        )
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name, 
            quantization_config=quant_config, 
            device_map="auto", max_memory={0: "12GB", "cpu": "16GB"}, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa"
        )
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model.eval()
        print("VLM loaded successfully!")

    def _get_device(self):
        """Returns the primary device of the model."""
        try:
            return next(self.model.parameters()).device
        except Exception:
            return torch.device("cpu")

    def answer_question(self, image_path: str, question: str) -> str:
        if self.model is None:
            self.load()
            
        try:
            image = Image.open(image_path).convert("RGB")
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question + "\n\nCRITICAL INSTRUCTION: You are a strict factual image analyzer. Base your answer ONLY on what is clearly and unambiguously visible in the image. If the specific object, action, or detail asked in the question is obscured, out of frame, not visible, or you are unsure, you MUST state that it is not visible or nothing is there. Under NO circumstances should you guess. For example, if asked what someone is holding and the hand is empty or not visible, answer: \"Not holding anything\"."}
                    ]
                }
            ]
            
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            inputs = self.processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt"
            ).to("cuda" if torch.cuda.is_available() else "cpu")
            
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=128)
                
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            return output_text.strip()
            
        except Exception as e:
            return f"Error during VLM inference: {str(e)}"
            
    def extract_events(self, video_desc: str) -> list[str]:
        if self.model is None:
            self.load()
            
        try:
            prompt = (
                f"Extract a chronological list of discrete events from the following video description. "
                f"Output ONLY a comma-separated list of the events in English, with no other text. "
                f"Video description: {video_desc}"
            )
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            inputs = self.processor(
                text=[text],
                padding=True,
                return_tensors="pt"
            ).to("cuda" if torch.cuda.is_available() else "cpu")
            
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=128)
                
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            # Parse the comma-separated output
            events = [e.strip() for e in output_text.split(",") if e.strip()]
            return events
            
        except Exception as e:
            print(f"Error during event extraction: {str(e)}")
            return []

    def _text_only_generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        """Internal: text-only inference using the loaded VLM (no image)."""
        if self.model is None:
            self.load()
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], padding=True, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

    def analyze_vqa_query(self, english_question: str) -> dict:
        """
        Improvements #7, #8, #11, #12 (VQA):
        Single Qwen2-VL text-only call with interleaved CoT JSON output.
        Returns: retrieval_description, vlm_question, paraphrases, objects_required,
                 needs_fact_lookup, fact_query.
        """
        import json, re

        prompt = (
            "You are a video retrieval assistant. Analyze the user question and output ONLY valid JSON.\n\n"
            "EXAMPLE:\n"
            "Question: \"What is the man in the red shirt holding?\"\n"
            "{\n"
            '  "fact_reasoning": "No external lookup needed, purely visual.",\n'
            '  "needs_fact_lookup": false,\n'
            '  "fact_query": "",\n'
            '  "retrieval_description": "A man wearing a red shirt",\n'
            '  "vlm_question": "What object is this man holding in his hands?",\n'
            '  "paraphrase_reasoning": "Vary the description to increase recall.",\n'
            '  "paraphrases": ["person in red clothing holding something", "man with red top grasping an object"],\n'
            '  "object_reasoning": "Query mentions a person and a shirt.",\n'
            '  "objects_required": ["Person", "Shirt"]\n'
            "}\n\n"
            "EXAMPLE 2:\n"
            "Question: \"Which country has the highest GDP according to this news?\"\n"
            "{\n"
            '  "fact_reasoning": "Highest GDP country needs external lookup for accuracy.",\n'
            '  "needs_fact_lookup": true,\n'
            '  "fact_query": "which country has the highest GDP in the world",\n'
            '  "retrieval_description": "news broadcast about economy and GDP rankings",\n'
            '  "vlm_question": "Which country is mentioned as having the highest GDP?",\n'
            '  "paraphrase_reasoning": "Search for economic news content.",\n'
            '  "paraphrases": ["economic rankings news segment", "GDP world ranking broadcast"],\n'
            '  "object_reasoning": "No specific objects to filter.",\n'
            '  "objects_required": []\n'
            "}\n\n"
            f'Now analyze:\nQuestion: "{english_question}"\n'
            "Output ONLY the JSON object, no other text."
        )

        raw = self._text_only_generate(prompt, max_new_tokens=400)

        # Robust JSON extraction (handle surrounding text)
        try:
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception:
            pass

        # Fallback: return safe defaults so pipeline continues
        print(f"[VLM] analyze_vqa_query JSON parse failed. Raw: {raw[:200]}")
        return {
            "needs_fact_lookup": False,
            "fact_query": "",
            "retrieval_description": english_question,
            "vlm_question": english_question,
            "paraphrases": [],
            "objects_required": [],
        }

    def verify_frame(self, image_path: str, retrieval_description: str) -> bool:
        """
        Improvement #5 (VQA): Visual yes/no verification.
        Returns True if the frame visually matches the retrieval description.
        """
        if self.model is None:
            self.load()

        prompt = (
            f"Does this image match the description: '{retrieval_description}'?\n"
            "Answer with ONLY 'yes' or 'no'."
        )
        try:
            image = Image.open(image_path).convert("RGB")
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=5)
            trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
            answer = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip().lower()
            return answer.startswith("yes")
        except Exception as e:
            print(f"[VLM] verify_frame error: {e}")
            return True  # Default to keeping the frame if verification fails






