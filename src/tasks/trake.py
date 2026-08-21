"""Video-first TRAKE retrieval with partial-order sequence alignment."""

from __future__ import annotations

from typing import Any

from src import config
from src.online_pipeline.rank_fusion import fuse_rankings
from src.online_pipeline.sequence_alignment import align_event_candidates
from src.online_pipeline.video_evidence import fuse_video_rankings
from src.utils.translation import translate_vi_to_en


class TrakeTask:
    """Retrieve event evidence per video; impose only stated temporal edges."""

    def __init__(self, encoder, retriever, vlm_pipeline=None, *, media_retriever=None, ocr=None, frame_neighborhood=None):
        self.encoder = encoder
        self.retriever = retriever
        self.vlm_pipeline = vlm_pipeline
        self.media_retriever = media_retriever
        self.ocr = ocr
        self.frame_neighborhood = frame_neighborhood

    @staticmethod
    def _translate(text: str) -> str:
        return translate_vi_to_en(text)

    def _search_global(self, texts: list[str], top_k: int) -> list[dict[str, Any]]:
        values: list[list[dict[str, Any]]] = []
        encode_batch = getattr(self.encoder, "encode_text_batch", None)
        vectors = encode_batch(texts) if callable(encode_batch) else [self.encoder.encode_text(text) for text in texts]
        search_batch = getattr(self.retriever, "search_batch", None)
        if callable(search_batch):
            values = search_batch(vectors, top_k=top_k)
        else:
            values = [self.retriever.search(vector, top_k=top_k) for vector in vectors]
        return fuse_rankings([(f"query_{index}", rows) for index, rows in enumerate(values)], rrf_k=config.KIS_RRF_K)

    def _local_event(self, vector, video_id: str, event_top_k: int) -> list[dict[str, Any]]:
        batch = getattr(self.retriever, "search_in_video_batch", None)
        if callable(batch):
            rows = batch(vector, video_id=video_id, top_k=event_top_k)
            return rows[0] if rows else []
        try:
            return self.retriever.search_in_video(vector, video_id=video_id, top_k=event_top_k)
        except AttributeError:
            return [item for item in self.retriever.search(vector, top_k=max(100, event_top_k)) if str(item.get("video_id")) == video_id][:event_top_k]

    def _analysis(self, description: str, events: list[str]) -> dict[str, Any]:
        default = {"retrieval_queries": [], "event_queries": [[] for _ in events], "temporal_edges": []}
        if self.vlm_pipeline is None:
            return default
        try:
            raw = self.vlm_pipeline.analyze_trake_query(description, events)
            if not isinstance(raw, dict):
                return default
            event_queries = raw.get("event_queries", [])
            normalised = []
            for index in range(len(events)):
                row = event_queries[index] if isinstance(event_queries, list) and index < len(event_queries) else []
                normalised.append([str(value).strip() for value in row if str(value).strip()] if isinstance(row, list) else [])
            edges = []
            for edge in raw.get("temporal_edges", []):
                if isinstance(edge, (list, tuple)) and len(edge) == 2:
                    try:
                        before, after = int(edge[0]), int(edge[1])
                    except (TypeError, ValueError):
                        continue
                    if 0 <= before < len(events) and 0 <= after < len(events) and before != after and (before, after) not in edges:
                        edges.append((before, after))
            return {
                "retrieval_queries": [str(value).strip() for value in raw.get("retrieval_queries", []) if str(value).strip()],
                "event_queries": normalised,
                "temporal_edges": edges,
            }
        except Exception as exc:
            print(f"[TRAKE] Qwen analysis unavailable; use parsed event text only: {exc}")
            return default

    def _qwen_event_rank(self, candidates: list[dict[str, Any]], event: str, budget: int) -> list[dict[str, Any]]:
        if self.vlm_pipeline is None:
            return []
        output: list[dict[str, Any]] = []
        score_fn = getattr(self.vlm_pipeline, "score_event_match_details", None)
        if not callable(score_fn):
            return []
        for input_rank, candidate in enumerate(candidates[:budget], start=1):
            try:
                details = score_fn(str(config.keyframe_path(candidate["video_id"], candidate["keyframe_name"])), event)
            except Exception as exc:
                print(f"[TRAKE] Qwen event score skipped: {exc}")
                continue
            if not details or details.get("score") is None:
                continue
            item = candidate.copy()
            item["qwen_event_score"] = int(details["score"])
            item["qwen_event_evidence"] = list(details.get("visible_evidence") or [])
            item["qwen_event_input_rank"] = input_rank
            output.append(item)
        return sorted(output, key=lambda item: (-int(item["qwen_event_score"]), int(item["qwen_event_input_rank"])))

    def execute(
        self,
        video_desc: str,
        events: list[str] | tuple[str, ...],
        top_videos: int = 5,
        event_top_k: int = config.TRAKE_EVENT_TOP_K,
        max_sequences: int = 100,
        beam_size: int = 50,
        qwen_per_event: int = config.TRAKE_QWEN_PER_EVENT,
        neighborhood_count: int = config.FRAME_NEIGHBORHOOD_COUNT,
    ) -> list[dict]:
        description = self._translate(video_desc)
        events = [str(event).strip() for event in events if str(event).strip()]
        auto_extracted = False
        if not events and self.vlm_pipeline:
            events = [event.strip() for event in self.vlm_pipeline.extract_events(video_desc) if event.strip()]
            auto_extracted = True
        if not events:
            return []
        english_events = [self._translate(event) for event in events]
        analysis = self._analysis(description, english_events)

        # Description and every event vote independently for target video.
        global_queries = [description, *english_events, *analysis["retrieval_queries"]]
        global_frames = self._search_global(list(dict.fromkeys(global_queries)), top_k=max(1000, top_videos * 100))
        video_sources: list[tuple[str, list[dict[str, Any]]]] = [("clip_description_events", global_frames)]
        if self.media_retriever is not None:
            try:
                media: list[dict[str, Any]] = []
                for query in [description, *analysis["retrieval_queries"]]:
                    for rank, item in enumerate(self.media_retriever.search(query, top_k=top_videos * 4), start=1):
                        value = item.copy()
                        value["video_id"] = str(value["video_id"]).removesuffix(".mp4")
                        value["score"] = float(value.get("metadata_score", 0.0))
                        value["metadata_rank"] = rank
                        media.append(value)
                if media:
                    video_sources.append(("btc_media_e5", media))
            except Exception as exc:
                print(f"[TRAKE] Media E5 unavailable: {exc}")
        videos = fuse_video_rankings(video_sources, rrf_k=config.KIS_RRF_K)[:top_videos]

        vectors = [self.encoder.encode_text(event) for event in english_events]
        results: list[dict[str, Any]] = []
        for video in videos:
            video_id = str(video["video_id"])
            event_lists: list[list[dict[str, Any]]] = []
            for event_index, (event, vector) in enumerate(zip(english_events, vectors)):
                base = self._local_event(vector, video_id, event_top_k)
                variants = analysis["event_queries"][event_index]
                if variants:
                    variant_rows = self._search_global(variants, top_k=max(100, event_top_k * 4))
                    variant_rows = [item for item in variant_rows if str(item.get("video_id")).removesuffix(".mp4") == video_id][:event_top_k]
                    if variant_rows:
                        base = fuse_rankings([("event_original", base), ("event_qwen_queries", variant_rows)], rrf_k=config.KIS_RRF_K)
                if self.ocr is not None and base:
                    try:
                        ocr_rows = self.ocr.rank(
                            event,
                            base,
                            lambda item: config.keyframe_path(item["video_id"], item["keyframe_name"]),
                        )
                        if ocr_rows:
                            base = fuse_rankings([("event_clip", base), ("event_ocr", ocr_rows)], rrf_k=config.KIS_RRF_K)
                    except Exception as exc:
                        print(f"[TRAKE] Candidate OCR unavailable: {exc}")
                qwen = self._qwen_event_rank(base, event, qwen_per_event)
                if qwen:
                    base = fuse_rankings([("event_clip", base), ("event_qwen", qwen)], rrf_k=config.KIS_RRF_K)
                event_lists.append(base[:event_top_k])
            chains = align_event_candidates(
                event_lists,
                temporal_edges=analysis["temporal_edges"],
                beam_size=beam_size,
                max_sequences=max_sequences,
            )
            for chain in chains:
                results.append({
                    "video_id": video_id,
                    "video_rank": video["video_rank"],
                    "video_rrf_score": video["rrf_score"],
                    "video_source_ranks": video["video_source_ranks"],
                    "sequence_score": float(chain["sequence_score"]) + float(video["rrf_score"]),
                    "events": chain["events"],
                    "temporal_edges": analysis["temporal_edges"],
                    "neighborhood_sequence_rank": 1,
                    "is_valid_sequence": len(chain["events"]) == len(events),
                    "auto_extracted_events": events if auto_extracted else None,
                })
        results.sort(key=lambda result: (-float(result["sequence_score"]), int(result.get("video_rank", 999)), int(result.get("neighborhood_sequence_rank", 1))))
        # Hedge the single strongest evidence chain, rather than spending an
        # exponential neighbourhood budget around every beam candidate.
        if self.frame_neighborhood is not None and results:
            best = results[0]
            proposals = self.frame_neighborhood.sequence_proposals(
                best["events"],
                edges=analysis["temporal_edges"],
                limit=max_sequences,
                count=neighborhood_count,
            )
            hedges = []
            for rank, proposal in enumerate(proposals, start=1):
                item = best.copy()
                item["events"] = proposal
                item["neighborhood_sequence_rank"] = rank
                hedges.append(item)
            merged, seen = [], set()
            for item in [*hedges, *results]:
                identity = (str(item["video_id"]), *(int(event["frame_id"]) for event in item["events"]))
                if identity not in seen:
                    seen.add(identity)
                    merged.append(item)
            results = merged
        return results[:max_sequences]
