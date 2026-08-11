import sys
from pathlib import Path
from deep_translator import GoogleTranslator
import re

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

class VQATask:
    def __init__(self, encoder, retriever, object_filter=None, vlm_pipeline=None):
        self.encoder = encoder
        self.retriever = retriever
        self.object_filter = object_filter
        self.vlm_pipeline = vlm_pipeline
    

    def execute(self, question: str, top_k: int = 5) -> list[dict]:
        try:
            english_query = GoogleTranslator(source='vi', target='en').translate(question)
        except Exception as e:
            print(f"Translation failed: {e}")
            english_query = question
            
        vector = self.encoder.encode_text(english_query)
        candidates = self.retriever.search(query_vector=vector, top_k=top_k)
        
        req_objs = []
        if self.object_filter:
            all_labels = self.object_filter.get_all_labels()
            query_lower = english_query.lower()
            for label in all_labels:
                pattern = r'\b' + re.escape(label.lower()) + r'(?:s|es)?\b'
                if re.search(pattern, query_lower):
                    req_objs.append(label)
                    
        if req_objs and self.object_filter:
            candidates = self.object_filter.filter_candidates(candidates, req_objs, mode="boost")
                
        for c in candidates:
            if self.vlm_pipeline:
                try:
                    c['answer'] = self.vlm_pipeline.answer_question(c['keyframe_path'], question)
                except Exception as e:
                    c['answer'] = f"Failed to run VLM: {e}"
            else:
                c['answer'] = "[Manual Review Required - No VLM loaded]"
                
        return candidates
