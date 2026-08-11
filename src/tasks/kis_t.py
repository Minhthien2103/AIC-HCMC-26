import sys
from pathlib import Path
import re
from deep_translator import GoogleTranslator

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.online_pipeline.temporal_search import TemporalSearchEngine

class KIStask:
    def __init__(self, encoder, retriever, object_filter = None):
        self.encoder = encoder
        self.retriever = retriever
        self.object_filter = object_filter
        self.temporal_engine = TemporalSearchEngine(encoder, retriever, object_filter)
        
    def execute(self, query: str, top_k: int = 100, object_labels: str = "") -> list[dict]:
        try:
            english_query = GoogleTranslator(source = 'vi', target = 'en').translate(query)
            print(f"Translated query: {english_query}")

        except Exception as e:
            print(f"Translation failed: {e}")
            english_query = query
            
        vector = self.encoder.encode_text(english_query)

        if self.temporal_engine.has_temporal_keywords(english_query):
            candidates = self.temporal_engine.execute(english_query, top_k = top_k)

            if not candidates:
                candidates = self.retriever.search(query_vector = vector, top_k = top_k)
        else:
            candidates = self.retriever.search(query_vector = vector, top_k = top_k)

        req_objs = []
        if object_labels:
            req_objs = [o.strip() for o in object_labels.split(",") if o.strip()]

        elif self.object_filter:
            all_labels = self.object_filter.get_all_labels()
            query_lower = english_query.lower()
            
            for label in all_labels:
                pattern = r'\b' + re.escape(label.lower()) + r'(?:s|es)?\b'

                if re.search(pattern, query_lower):
                    req_objs.append(label)
                    
        if req_objs and self.object_filter:
            candidates = self.object_filter.filter_candidates(candidates, req_objs, mode="boost")

            for c in candidates:
                if "auto_extracted" not in c:
                    c["auto_extracted"] = req_objs
            
        return candidates
