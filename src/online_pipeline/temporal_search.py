import sys
import re
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.online_pipeline.sequence_alignment import align_event_candidates

class TemporalSearchEngine:
    def __init__(self, encoder, retriever, object_filter=None):
        self.encoder = encoder
        self.retriever = retriever
        self.object_filter = object_filter
        
        # Sort keywords by length (longest first) to prevent partial matching (e.g., "sau do" vs "sau")
        self.keywords = sorted(config.TEMPORAL_KEYWORDS, key=len, reverse=True)
        
    def has_temporal_keywords(self, query: str) -> bool:
        """Returns True if the query contains any temporal keywords."""
        query_lower = query.lower()
        pattern = r'\b(?:' + '|'.join(map(re.escape, self.keywords)) + r')\b'
        return bool(re.search(pattern, query_lower))
        
    def _split_query(self, query: str) -> list[str]:
        """Splits the query into sub-events based on temporal keywords."""
        pattern = r'\b(?:' + '|'.join(map(re.escape, self.keywords)) + r')\b'
        sub_events = [s.strip() for s in re.split(pattern, query, flags=re.IGNORECASE) if s.strip()]
        return sub_events

    def execute(self, query: str, top_k: int = 100) -> list[dict]:
        sub_events = self._split_query(query)
        
        if len(sub_events) < 2:
            print("Temporal search triggered, but less than 2 sub-events found. Falling back.")
            return []
            
        print(f"Temporal Search splitting query into {len(sub_events)} events: {sub_events}")
            
        event_results = []
        for i, event in enumerate(sub_events):
            vec = self.encoder.encode_text(event)
            # Retrieve candidates for each sub-event
            candidates = self.retriever.search(vec, top_k=top_k)
            event_results.append(candidates)
            
        # Group all candidates by video_id
        video_map = {}
        for ev_idx, candidates in enumerate(event_results):
            for c in candidates:
                vid = c["video_id"]
                if vid not in video_map:
                    video_map[vid] = [[] for _ in range(len(sub_events))]
                
                video_map[vid][ev_idx].append(c)
                
        final_candidates = []
        
        # Find valid chronological sequences within the same video.  The
        # alignment helper supports any number of sub-events and never emits a
        # partial chain.
        for vid, event_lists in video_map.items():
            # If the video doesn't contain matches for ALL sub-events, skip it
            if any(len(lst) == 0 for lst in event_lists):
                continue
                
            chains = align_event_candidates(event_lists, beam_size=max(10, top_k), max_sequences=1)
            if chains:
                chain = chains[0]
                res = chain["events"][0].copy()
                res["score"] = chain["sequence_score"]
                res["matched_sequence"] = [c["frame_id"] for c in chain["events"]]
                final_candidates.append(res)
                    
        # Sort the final videos by the strongest valid temporal chain found
        final_candidates.sort(key=lambda x: x["score"], reverse=True)
        
        # Tag them for the UI
        for c in final_candidates:
            c["is_temporal"] = True
            
        return final_candidates
