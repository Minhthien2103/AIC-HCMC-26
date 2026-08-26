import os
from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai
from PIL import Image, ImageDraw
import logging

LOGGER = logging.getLogger(__name__)

class GeminiPipeline:
    def __init__(self, model_name: str = "gemini-3.6-flash"):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            LOGGER.warning("GEMINI_API_KEY not found in environment!")
        else:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(model_name)
    
    def _create_collage(self, frame_paths: list[str], labels: list[str]) -> Image.Image:
        images = []
        for p in frame_paths:
            try:
                images.append(Image.open(p).convert("RGB"))
            except Exception:
                images.append(Image.new("RGB", (320, 240), color="black"))
        
        target_height = 240
        resized = []
        for img in images:
            w, h = img.size
            new_w = int(w * (target_height / h))
            resized.append(img.resize((new_w, target_height)))
            
        total_width = sum(img.size[0] for img in resized)
        collage = Image.new("RGB", (total_width, target_height + 30), color="white")
        
        x_offset = 0
        draw = ImageDraw.Draw(collage)
        for i, img in enumerate(resized):
            collage.paste(img, (x_offset, 30))
            draw.text((x_offset + 10, 10), labels[i], fill=(0, 0, 0))
            x_offset += img.size[0]
            
        return collage

    def evaluate_kis_candidates(self, query: str, candidate_videos: list[dict]) -> str | None:
        if not candidate_videos or not self.api_key:
            return None
            
        row_images = []
        for i, cand in enumerate(candidate_videos):
            vid = cand["video_id"]
            frames = cand["frames"][:3]
            row_labels = [f"Option {i+1}: {vid}"] + [""] * (len(frames) - 1)
            row_img = self._create_collage(frames, row_labels)
            row_images.append(row_img)
            
        total_height = sum(img.size[1] for img in row_images)
        max_width = max(img.size[0] for img in row_images)
        mega_collage = Image.new("RGB", (max_width, total_height), color="white")
        y_offset = 0
        for img in row_images:
            mega_collage.paste(img, (0, y_offset))
            y_offset += img.size[1]
            
        prompt = f"""
        I am looking for a video that exactly matches this chronological sequence of events:
        "{query}"
        
        I have provided a storyboard collage containing several Candidate Options.
        Each Option shows chronological keyframes from a different video.
        
        Analyze all the Options. Which Option (video) is the BEST match for the sequence described in the text?
        You MUST respond with exactly the Video ID of the winner, and nothing else.
        For example, if Option 2 is the best, just output its Video ID (e.g. L30_046).
        If none of them match, output "NONE".
        """
        
        try:
            response = self.model.generate_content([mega_collage, prompt])
            ans = response.text.strip()
            LOGGER.info(f"[Gemini] Rerank winner: {ans}")
            return ans
        except Exception as e:
            LOGGER.error(f"Gemini API error: {e}")
            return None

    # --- VQA Compatibility Methods to save API calls ---
    def analyze_vqa_query(self, english_question: str) -> dict:
        return {} # Disable to save API limit

    def _text_only_generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        return "" # Disable

    def verify_frame(self, image_path: str, retrieval_description: str) -> bool:
        return True # Assume True to save API limit

    def read_text_in_image(self, image_path: str, question: str, lang: str = "vi") -> str:
        return self.answer_question(image_path, question)

    def answer_question(self, image_path: str, question: str) -> str:
        if not self.api_key:
            return "N/A"
        try:
            img = Image.open(image_path).convert("RGB")
            prompt = f"Answer this question based on the text visible in the image: {question}. Keep it very brief."
            response = self.model.generate_content([img, prompt])
            ans = response.text.strip()
            LOGGER.info(f"[Gemini] OCR Answer: {ans}")
            return ans
        except Exception as e:
            LOGGER.error(f"Gemini OCR error: {e}")
            return "N/A"

    def extract_events(self, video_desc: str) -> list[str]:
        """For TRAKE parser fallback."""
        if not self.api_key:
            return []
        try:
            prompt = f"Extract exactly 3 chronological events from this text: '{video_desc}'. Format as a python list of strings."
            response = self.model.generate_content(prompt)
            # basic parse
            return [line.strip("- *'\"") for line in response.text.split("\n") if len(line) > 5][:3]
        except:
            return []