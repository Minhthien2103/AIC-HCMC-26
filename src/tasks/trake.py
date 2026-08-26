from src.tasks.gemini_api import GeminiPipeline
gemini_api = GeminiPipeline()

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.online_pipeline.sequence_alignment import align_event_candidates
from src.utils.translation import translate_vi_to_en


class TrakeTask:
    """Retrieve complete chronological event sequences for TRAKE."""

    def __init__(self, encoder, retriever, vlm_pipeline=None):
        self.encoder = encoder
        self.retriever = retriever
        self.vlm_pipeline = vlm_pipeline

    @staticmethod
    def _translate(text: str) -> str:
        return translate_vi_to_en(text)

    def _qwen_rerank_sequences(self, chains: list[dict], events: list[str]) -> list[dict]:
        from src import config
        chains.sort(key=lambda x: x["sequence_score"], reverse=True)
        top_chains = chains[:10]
        other_chains = chains[10:]
        
        candidate_payload = []
        for chain in top_chains:
            vid = chain["video_id"]
            chain_frames = chain["events"]
            img_paths = []
            for i, ev in enumerate(events):
                if i < len(chain_frames):
                    fid = chain_frames[i].get("frame_idx", chain_frames[i].get("frame_id"))
                    img_paths.append(str(config.keyframe_path(vid, str(fid).zfill(4))))
            candidate_payload.append({"video_id": vid, "frames": img_paths})
            
        print(f"[TRAKE Gemini] Sending {len(candidate_payload)} candidates for sequence verification...")
        query_text = " THEN ".join(events)
        winner_vid = gemini_api.evaluate_kis_candidates(query_text, candidate_payload)
        
        if winner_vid and winner_vid != "NONE":
            print(f"[TRAKE Gemini] Winner selected: {winner_vid}")
            for chain in top_chains:
                if chain["video_id"] == winner_vid:
                    chain["sequence_score"] += 1000.0
                    break
                    
        top_chains.sort(key=lambda x: x["sequence_score"], reverse=True)
        return top_chains + other_chains

    def execute(
        self,
        video_desc: str,
        events: list[str] | tuple[str, ...],
        top_videos: int = 5,
        event_top_k: int = 20,
        max_sequences: int = 100,
        beam_size: int = 50,
    ) -> list[dict]:
        eng_video_desc = self._translate(video_desc)
        events = [str(event).strip() for event in events if str(event).strip()]
        auto_extracted = False
        if not events and self.vlm_pipeline:
            events = [event.strip() for event in self.vlm_pipeline.extract_events(video_desc) if event.strip()]
            auto_extracted = True
        if not events:
            return []

        eng_events = [self._translate(event) for event in events]
        search_queries = [eng_video_desc] + eng_events
        
        video_rrf_scores: dict[str, float] = {}
        video_max_scores: dict[str, float] = {}
        
        for query in search_queries:
            vector = self.encoder.encode_text(query)
            wide_candidates = self.retriever.search(vector, top_k=max(100, top_videos * 100))
            
            query_video_scores = {}
            for candidate in wide_candidates:
                vid = candidate["video_id"]
                score = float(candidate.get("score", 0.0))
                query_video_scores[vid] = max(query_video_scores.get(vid, float("-inf")), score)
                video_max_scores[vid] = max(video_max_scores.get(vid, float("-inf")), score)
                
            sorted_vids_for_query = sorted(query_video_scores.keys(), key=lambda v: query_video_scores[v], reverse=True)
            for rank, vid in enumerate(sorted_vids_for_query):
                video_rrf_scores[vid] = video_rrf_scores.get(vid, 0.0) + (1.0 / (60 + rank))
        
        sorted_videos = sorted(video_rrf_scores.keys(), key=lambda v: video_rrf_scores[v], reverse=True)[:top_videos]
        
        # Keep original logic compatibility by setting video_scores
        video_scores = video_max_scores

        results: list[dict] = []
        event_vectors = [self.encoder.encode_text(event) for event in eng_events]
        for vid in sorted_videos:
            event_lists = [
                self.retriever.search_in_video(vector, video_id=vid, top_k=event_top_k)
                for vector in event_vectors
            ]
            chains = align_event_candidates(event_lists, beam_size=beam_size, max_sequences=max_sequences)
            for chain in chains:
                results.append({
                    "video_id": vid,
                    "video_score": video_scores[vid],
                    "sequence_score": chain["sequence_score"] + video_scores[vid],
                    "events": chain["events"],
                    "is_valid_sequence": len(chain["events"]) == len(events),
                    "auto_extracted_events": events if auto_extracted else None,
                })

        results.sort(key=lambda result: result["sequence_score"], reverse=True)
        results = self._qwen_rerank_sequences(results, events)
        return results[:max_sequences]
