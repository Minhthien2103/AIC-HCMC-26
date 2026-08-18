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

        vector = self.encoder.encode_text(eng_video_desc)
        wide_candidates = self.retriever.search(vector, top_k=max(100, top_videos * 100))
        video_scores: dict[str, float] = {}
        for candidate in wide_candidates:
            vid = candidate["video_id"]
            video_scores[vid] = max(video_scores.get(vid, float("-inf")), float(candidate["score"]))
        sorted_videos = sorted(video_scores, key=video_scores.get, reverse=True)[:top_videos]

        results: list[dict] = []
        event_vectors = [self.encoder.encode_text(self._translate(event)) for event in events]
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
        return results[:max_sequences]
