import sys
from pathlib import Path
from deep_translator import GoogleTranslator

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

class TrakeTask:
    def __init__(self, encoder, retriever, vlm_pipeline = None):
        self.encoder = encoder
        self.retriever = retriever
        self.vlm_pipeline = vlm_pipeline
        
    def execute(self, video_desc: str, events: list[str], top_videos: int = 5) -> list[dict]:
        try:
            eng_video_desc = GoogleTranslator(source='vi', target='en').translate(video_desc)
        except Exception:
            eng_video_desc = video_desc
            
        auto_extracted = False
        if not events and self.vlm_pipeline:
            # Auto-extract events from description using VLM
            events = self.vlm_pipeline.extract_events(video_desc)
            auto_extracted = True
            
        # Phase A: Find target videos
        vector = self.encoder.encode_text(eng_video_desc)
        wide_candidates = self.retriever.search(vector, top_k=500)
        
        video_scores = {}
        for c in wide_candidates:
            vid = c["video_id"]
            if vid not in video_scores or c["score"] > video_scores[vid]["score"]:
                video_scores[vid] = c
                
        sorted_videos = sorted(video_scores.values(), key=lambda x: x["score"], reverse=True)[:top_videos]
        
        results = []
        # Phase B: Find events within target videos
        for vid_candidate in sorted_videos:
            vid = vid_candidate["video_id"]
            vid_events = []
            
            for event_desc in events:
                if not event_desc.strip():
                    continue
                try:
                    eng_event = GoogleTranslator(source='vi', target='en').translate(event_desc)
                except Exception:
                    eng_event = event_desc
                    
                e_vec = self.encoder.encode_text(eng_event)
                e_cands = self.retriever.search_in_video(e_vec, video_id=vid, top_k=1)
                if e_cands:
                    vid_events.append(e_cands[0])
                    
            # Basic temporal constraint verification
            valid_sequence = True
            for i in range(len(vid_events) - 1):
                if vid_events[i]["frame_id"] >= vid_events[i+1]["frame_id"]:
                    valid_sequence = False
                    break
                    
            results.append({
                "video_id": vid,
                "video_score": vid_candidate["score"],
                "events": vid_events,
                "is_valid_sequence": valid_sequence,
                "auto_extracted_events": events if auto_extracted else None
            })
            
        return results
