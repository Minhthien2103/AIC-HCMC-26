"""Video-first visual question answering."""

from __future__ import annotations

from typing import Any

from src import config
from src.online_pipeline.rank_fusion import candidate_identity, fuse_rankings
from src.online_pipeline.video_evidence import d_hondt_allocate, fuse_video_rankings, stratified_candidates
from src.utils.translation import translate_vi_to_en


class VQATask:
    """Select videos by ranked evidence before asking Qwen for an answer."""

    def __init__(self, encoder, retriever, object_filter=None, vlm_pipeline=None, semantic_filter=None, *, media_retriever=None, ocr=None, frame_neighborhood=None):
        self.encoder = encoder
        self.retriever = retriever
        self.object_filter = object_filter  # no longer boosts VQA ranking
        self.vlm_pipeline = vlm_pipeline
        self.semantic_filter = semantic_filter
        self.media_retriever = media_retriever
        self.ocr = ocr
        self.frame_neighborhood = frame_neighborhood

    @staticmethod
    def _unique(values: list[str]) -> list[str]:
        return list(dict.fromkeys(value.strip() for value in values if str(value).strip()))

    def _global_source(self, queries: list[str], top_k: int) -> list[dict[str, Any]]:
        encode_batch = getattr(self.encoder, "encode_text_batch", None)
        vectors = encode_batch(queries) if callable(encode_batch) else [self.encoder.encode_text(query) for query in queries]
        search_batch = getattr(self.retriever, "search_batch", None)
        values = search_batch(vectors, top_k=top_k) if callable(search_batch) else [self.retriever.search(vector, top_k=top_k) for vector in vectors]
        return fuse_rankings([(f"query_{index}", rows) for index, rows in enumerate(values)], rrf_k=config.VQA_RRF_K)

    def _local_source(self, queries: list[str], video_ids: list[str], budget: int) -> list[dict[str, Any]]:
        encode_batch = getattr(self.encoder, "encode_text_batch", None)
        vectors = encode_batch(queries) if callable(encode_batch) else [self.encoder.encode_text(query) for query in queries]
        lists = [[] for _ in vectors]
        for video_id in video_ids:
            try:
                batch = getattr(self.retriever, "search_in_video_batch", None)
                per_query = batch(vectors, video_id=video_id, top_k=budget) if callable(batch) else [self.retriever.search_in_video(vector, video_id, budget) for vector in vectors]
            except AttributeError:
                per_query = [[item for item in self.retriever.search(vector, top_k=max(100, budget)) if str(item.get("video_id")) == video_id][:budget] for vector in vectors]
            for index, rows in enumerate(per_query):
                lists[index].extend(rows)
        return fuse_rankings([(f"local_{index}", rows) for index, rows in enumerate(lists)], rrf_k=config.VQA_RRF_K)

    @staticmethod
    def _by_video(candidates: list[dict[str, Any]], ids: list[str]) -> dict[str, list[dict[str, Any]]]:
        grouped = {video_id: [] for video_id in ids}
        for candidate in candidates:
            video_id = str(candidate.get("video_id", "")).removesuffix(".mp4")
            if video_id in grouped:
                grouped[video_id].append(candidate)
        for rows in grouped.values():
            rows.sort(key=lambda item: (-float(item.get("score", 0.0)), candidate_identity(item)))
        return grouped

    def _analysis(self, question: str) -> dict[str, Any]:
        default = {"retrieval_description": question, "vlm_question": question, "paraphrases": [], "metadata_queries": [], "ocr_queries": []}
        if self.vlm_pipeline is None:
            return default
        try:
            raw = self.vlm_pipeline.analyze_vqa_query(question) or {}
            return {
                "retrieval_description": str(raw.get("retrieval_description") or question),
                "vlm_question": str(raw.get("vlm_question") or question),
                "paraphrases": self._unique([str(value) for value in raw.get("paraphrases", [])]),
                "metadata_queries": self._unique([str(value) for value in raw.get("metadata_queries", [])]),
                "ocr_queries": self._unique([str(value) for value in raw.get("ocr_queries", [])]),
            }
        except Exception as exc:
            print(f"[VQA] Qwen analysis unavailable: {exc}")
            return default

    def execute(self, question: str, top_k: int | None = None, progress_callback=None, allow_external_search: bool | None = None) -> tuple[list[dict], dict]:
        del allow_external_search
        top_k = config.VQA_RERANK_K if top_k is None else max(1, min(int(top_k), config.VQA_MAX_CANDIDATES))
        english = translate_vi_to_en(question)
        analysis = self._analysis(english)
        retrieval_queries = self._unique([english, analysis["retrieval_description"], *analysis["paraphrases"]])
        global_frames = self._global_source(retrieval_queries, top_k=max(1000, top_k * 10))
        video_sources: list[tuple[str, list[dict[str, Any]]]] = [("clip_vitb32", global_frames)]
        if self.media_retriever is not None:
            try:
                media = []
                for query in self._unique([english, *analysis["metadata_queries"]]):
                    for rank, item in enumerate(self.media_retriever.search(query, top_k=config.VQA_VIDEO_BUDGET * 4), start=1):
                        row = item.copy()
                        row["video_id"] = str(row["video_id"]).removesuffix(".mp4")
                        row["score"] = float(row.get("metadata_score", 0.0))
                        row["metadata_rank"] = rank
                        media.append(row)
                if media:
                    video_sources.append(("btc_media_e5", media))
            except Exception as exc:
                print(f"[VQA] Media E5 unavailable: {exc}")
        videos = fuse_video_rankings(video_sources, rrf_k=config.VQA_RRF_K)[:config.VQA_VIDEO_BUDGET]
        ids = [str(video["video_id"]) for video in videos]
        local = self._local_source(retrieval_queries, ids, config.VQA_LOCAL_FRAME_BUDGET)
        if self.ocr is not None:
            try:
                ocr_queries = self._unique([english, *analysis["ocr_queries"]])
                ocr_pool = stratified_candidates(
                    self._by_video(local, ids), videos, limit=config.KIS_OCR_CANDIDATE_BUDGET
                )
                ocr_lists = [
                    self.ocr.rank(text, ocr_pool, lambda item: config.keyframe_path(item["video_id"], item["keyframe_name"]))
                    for text in ocr_queries
                ]
                ocr_rows = fuse_rankings([(f"ocr_{index}", rows) for index, rows in enumerate(ocr_lists)], rrf_k=config.VQA_RRF_K)
                if ocr_rows:
                    local = fuse_rankings([("clip_local", local), ("candidate_ocr_e5", ocr_rows)], rrf_k=config.VQA_RRF_K)
            except Exception as exc:
                print(f"[VQA] Candidate OCR unavailable: {exc}")
        video_evidence = {str(video["video_id"]): video for video in videos}
        for candidate in local:
            evidence = video_evidence.get(str(candidate.get("video_id", "")).removesuffix(".mp4"))
            if evidence is not None:
                candidate["video_rank"] = int(evidence["video_rank"])
                candidate["video_rrf_score"] = float(evidence["rrf_score"])
                candidate["video_source_ranks"] = dict(evidence["video_source_ranks"])
                candidate["video_source_frame_ranks"] = dict(evidence["video_source_frame_ranks"])
            candidate.setdefault("anchor_frame_id", int(candidate["frame_id"]))
        grouped = self._by_video(local, ids)
        answer_pool = stratified_candidates(grouped, videos, limit=config.VQA_QWEN_CANDIDATE_BUDGET)
        answered: list[dict[str, Any]] = []
        for index, candidate in enumerate(answer_pool, start=1):
            item = candidate.copy()
            item["answer_error"] = ""
            details = None
            try:
                detail_fn = getattr(self.vlm_pipeline, "answer_question_details", None) if self.vlm_pipeline else None
                details = detail_fn(str(config.keyframe_path(item["video_id"], item["keyframe_name"])), analysis["vlm_question"]) if callable(detail_fn) else None
                if details is None and self.vlm_pipeline:
                    answer = self.vlm_pipeline.answer_question(str(config.keyframe_path(item["video_id"], item["keyframe_name"])), analysis["vlm_question"])
                    details = {"answer": answer, "visible_evidence": [], "confidence": 1}
            except Exception as exc:
                item["answer_error"] = str(exc)
            if not details:
                continue
            answer = str(details.get("answer") or "").strip()
            if not answer or int(details.get("confidence", 0)) <= 0:
                continue
            item["answer"] = answer
            item["visible_evidence"] = list(details.get("visible_evidence") or [])
            item["answer_confidence"] = int(details.get("confidence", 0))
            item["answer_input_rank"] = index
            answered.append(item)
            if progress_callback:
                progress_callback(len(answered), top_k)
        # Do not let a generic/not-visible fallback silently occupy rank one.
        answered.sort(key=lambda item: (-int(item["answer_confidence"]), -float(item.get("score", 0.0)), int(item["answer_input_rank"])))
        if self.frame_neighborhood is not None:
            answered = self.frame_neighborhood.expand_ranked(answered, count=config.FRAME_NEIGHBORHOOD_COUNT)
        return answered[:top_k], analysis
