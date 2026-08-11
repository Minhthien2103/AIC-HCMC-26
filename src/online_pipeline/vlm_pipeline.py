import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from PIL import Image

class VLMPipeline:
    def __init__(self, model_name="Qwen/Qwen2-VL-7B-Instruct"):
        self.model_name = model_name
        self.model = None
        self.processor = None
        
    def load(self):
        if self.model is not None:
            return
            
        print(f"Loading VLM {self.model_name} in 4-bit...")
        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name, 
            quantization_config=quant_config, 
            device_map="auto", 
            attn_implementation="sdpa"
        )
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model.eval()
        print("VLM loaded successfully!")

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
                        {"type": "text", "text": question}
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
            ).to("cuda")
            
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
            ).to("cuda")
            
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
