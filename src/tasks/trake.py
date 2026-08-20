import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.online_pipeline.sequence_alignment import align_event_candidates
from src.utils.translation import translate_vi_to_en

_RRF_K = 60


def _rrf_video_scores(ranked_lists: list, k: int = _RRF_K) -> dict[str, float]:
    """Fuse multiple FAISS result lists and return per-video RRF score."""
    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        video_seen_in_list: dict[str, int] = {}
        rank = 0
        for candidate in ranked_list:
            vid = candidate["video_id"]
            if vid not in video_seen_in_list:
                video_seen_in_list[vid] = rank
                scores[vid] = scores.get(vid, 0.0) + 1.0 / (k + rank + 1)
                rank += 1
    return scores


def _event_paraphrases(event_en: str) -> list[str]:
    """Generate simple keyword variants for an event query.
    Focuses on the core action verb + object to help CLIP match keyframes.
    """
    # Strip common TRAKE preambles like "Khoảnh khắc đầu tiên..."
    stripped = re.sub(
        r'^(the moment|first moment|moment when|kho[aả]nh kh[aắ]c)[^a-zA-Z]*',
        '', event_en, flags=re.IGNORECASE
    ).strip()
    variants = list(dict.fromkeys([event_en, stripped]))
    return [v for v in variants if v]


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
        top_videos: int = 10,
        event_top_k: int = 20,
        max_sequences: int = 100,
        beam_size: int = 50,
    ) -> list[dict]:
        # ── Step 1: Translate ────────────────────────────────────────────
        eng_video_desc = self._translate(video_desc)
        events = [str(event).strip() for event in events if str(event).strip()]
        auto_extracted = False
        if not events and self.vlm_pipeline:
            events = [e.strip() for e in self.vlm_pipeline.extract_events(video_desc) if e.strip()]
            auto_extracted = True
        if not events:
            return []

        eng_events = [self._translate(ev) for ev in events]

        # ── Step 2: Multi-query video ranking with RRF (#19) ─────────────
        # Use video description + each individual event to find candidate videos.
        # This prevents one dominant event from drowning out the others.
        candidate_pool = max(200, top_videos * 20)
        search_queries = [eng_video_desc] + eng_events
        ranked_lists = []
        for q in search_queries:
            vec = self.encoder.encode_text(q)
            res = self.retriever.search(vec, top_k=candidate_pool)
            if res:
                ranked_lists.append(res)

        video_scores = _rrf_video_scores(ranked_lists)
        sorted_videos = sorted(video_scores, key=video_scores.get, reverse=True)[:top_videos]
        print(f"[TRAKE] Top videos: {sorted_videos[:5]}")

        # ── Step 3: Per-event search with paraphrase variants (#19) ──────
        # For each event use multiple CLIP queries and take the best frame.
        results: list[dict] = []
        for vid in sorted_videos:
            event_lists = []
            for ev_en in eng_events:
                variants = _event_paraphrases(ev_en)
                # Search each variant and merge per-video results; keep best score per frame
                frame_best: dict[int, dict] = {}
                for variant in variants:
                    vec = self.encoder.encode_text(variant)
                    hits = self.retriever.search_in_video(vec, video_id=vid, top_k=event_top_k)
                    for h in hits:
                        fid = h["faiss_idx"]
                        if fid not in frame_best or h["score"] > frame_best[fid]["score"]:
                            frame_best[fid] = h
                event_lists.append(sorted(frame_best.values(), key=lambda x: x["score"], reverse=True)[:event_top_k])

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

        results.sort(key=lambda r: r["sequence_score"], reverse=True)
        return results[:max_sequences]