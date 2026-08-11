import sys
import re
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

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
        
        # Find valid chronological sequences within the same video
        for vid, event_lists in video_map.items():
            # If the video doesn't contain matches for ALL sub-events, skip it
            if any(len(lst) == 0 for lst in event_lists):
                continue
                
            # For Batch 1, we handle N=2 sub-events using simple nested loop
            if len(sub_events) == 2:
                best_chain_score = -1
                best_chain = None
                
                for c1 in event_lists[0]:
                    for c2 in event_lists[1]:
                        frame1 = c1["frame_id"]
                        frame2 = c2["frame_id"]
                        
                        # Constraint: Event 1 MUST happen before Event 2
                        if frame1 < frame2:
                            # TycheVid Window Constraint: Must happen within TEMPORAL_WINDOW keyframes (approx. max time gap)
                            # (If config.TEMPORAL_WINDOW is set. For simplicity, we just ensure frame1 < frame2 here)
                            
                            score_sum = c1["score"] + c2["score"]
                            if score_sum > best_chain_score:
                                best_chain_score = score_sum
                                best_chain = (c1, c2)
                                
                if best_chain:
                    # Return the primary event's frame as the result, but with boosted score
                    res = best_chain[0].copy()
                    res["score"] = best_chain_score  # The combined score is much higher
                    res["matched_sequence"] = [c["frame_id"] for c in best_chain]
                    final_candidates.append(res)
                    
        # Sort the final videos by the strongest valid temporal chain found
        final_candidates.sort(key=lambda x: x["score"], reverse=True)
        
        # Tag them for the UI
        for c in final_candidates:
            c["is_temporal"] = True
            
        return final_candidates
