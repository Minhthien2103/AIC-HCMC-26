"""Video-first known-item search with auditable multi-source evidence."""

from __future__ import annotations

import re
import time
from typing import Any

from src import config
from src.online_pipeline.rank_fusion import candidate_identity, fuse_rankings
from src.online_pipeline.video_evidence import d_hondt_allocate, fuse_video_rankings, stratified_candidates
from src.utils.translation import translate_vi_to_en


_SENTENCE_BOUNDARY_RE = re.compile(r"(?:\n+|(?<=[.!?])\s+)")
_CLAUSE_BOUNDARY_RE = re.compile(r"(?<=[,;:])\s+|(?<=[–—])\s+")


class KIStask:
    """Known-item search: retrieve videos first, then localise their frames.

    Every retriever contributes one ranked list. Scores from CLIP, E5 and
    Qwen are never added directly: their ranks are fused with equal RRF.
    """

    def __init__(
        self,
        encoder,
        retriever,
        object_filter=None,
        vlm_pipeline=None,
        *,
        enable_qwen: bool = False,
        vlm_top_k: int | None = None,
        secondary_retrievers: dict[str, tuple[Any, Any]] | None = None,
        media_retriever=None,
        ocr=None,
        evidence_cache=None,
        candidate_budget: int | None = None,
        ocr_candidate_budget: int | None = None,
        text_frames_per_video: int | None = None,
        frame_neighborhood=None,
        video_budget: int | None = None,
        local_frame_budget: int | None = None,
        query_variant_limit: int | None = None,
        neighborhood_count: int | None = None,
        strict_sources: bool = False,
    ):
        self.encoder = encoder
        self.retriever = retriever
        self.object_filter = object_filter  # compatibility only; no KIS label rules
        self.vlm_pipeline = vlm_pipeline
        self.enable_qwen = bool(enable_qwen and vlm_pipeline is not None)
        self.vlm_top_k = int(vlm_top_k or config.KIS_QWEN_RERANK_TOP_K)
        self.secondary_retrievers = dict(secondary_retrievers or {})
        self.media_retriever = media_retriever
        self.ocr = ocr
        self.evidence_cache = evidence_cache
        self.candidate_budget = int(candidate_budget or config.KIS_DUAL_CANDIDATE_BUDGET)
        self.ocr_candidate_budget = int(ocr_candidate_budget or config.KIS_OCR_CANDIDATE_BUDGET)
        self.local_frame_budget = int(local_frame_budget or text_frames_per_video or config.KIS_LOCAL_FRAME_BUDGET)
        self.video_budget = int(video_budget or config.KIS_VIDEO_BUDGET)
        self.query_variant_limit = int(query_variant_limit or config.KIS_QUERY_VARIANT_LIMIT)
        self.neighborhood_count = int(neighborhood_count or config.FRAME_NEIGHBORHOOD_COUNT)
        self.frame_neighborhood = frame_neighborhood
        self.strict_sources = strict_sources

    def _context_limit(self, encoder=None) -> int:
        return max(4, int(getattr(encoder or self.encoder, "text_context_length", 77)))

    def _token_count(self, text: str, encoder=None) -> int:
        counter = getattr(encoder or self.encoder, "text_token_count", None)
        return int(counter(text)) if callable(counter) else len(re.findall(r"\S+", text)) + 2

    def _split_oversized_fragment(self, fragment: str, budget: int, encoder=None) -> list[str]:
        if self._token_count(fragment, encoder) <= budget:
            return [fragment]
        clauses = [part.strip() for part in _CLAUSE_BOUNDARY_RE.split(fragment) if part.strip()]
        if len(clauses) > 1:
            return [piece for clause in clauses for piece in self._split_oversized_fragment(clause, budget, encoder)]
        words, pieces, current = fragment.split(), [], []
        for word in words:
            candidate = " ".join([*current, word])
            if current and self._token_count(candidate, encoder) > budget:
                pieces.append(" ".join(current))
                current = [word]
            else:
                current.append(word)
        if current:
            pieces.append(" ".join(current))
        return pieces

    def _query_variants(self, query: str, encoder=None) -> list[str]:
        """Split overlength CLIP text without dropping its final constraint."""
        normalized = re.sub(r"\s+", " ", query).strip()
        if not normalized:
            return []
        budget = self._context_limit(encoder)
        if self._token_count(normalized, encoder) <= budget:
            return [normalized]
        fragments = [part.strip() for part in _SENTENCE_BOUNDARY_RE.split(normalized) if part.strip()]
        atomic = [piece for fragment in fragments for piece in self._split_oversized_fragment(fragment, budget, encoder) if piece]
        variants, current = [], ""
        for fragment in atomic:
            candidate = f"{current} {fragment}".strip()
            if current and self._token_count(candidate, encoder) > budget:
                variants.append(current)
                current = fragment
            else:
                current = candidate
        if current:
            variants.append(current)
        return list(dict.fromkeys(variants))

    @staticmethod
    def _unique_strings(values: list[str], excluded: set[str] | None = None) -> list[str]:
        seen = set(excluded or set())
        output: list[str] = []
        for value in values:
            normalized = re.sub(r"\s+", " ", str(value)).strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                output.append(normalized)
        return output

    @staticmethod
    def _canonical_video_id(value: object) -> str:
        return str(value).strip().removesuffix(".mp4")

    def _candidate_pool_size(self, top_k: int, retriever=None) -> int:
        available = getattr(getattr(retriever or self.retriever, "index", None), "ntotal", None)
        amount = max(int(top_k), self.candidate_budget)
        return min(amount, int(available)) if available is not None else amount

    def _search_variants(self, encoder, retriever, variants: list[str], pool_size: int, source_name: str) -> list[list[dict[str, Any]]]:
        if not variants:
            return []
        started = time.perf_counter()
        encode_batch = getattr(encoder, "encode_text_batch", None)
        vectors = encode_batch(variants) if callable(encode_batch) else [encoder.encode_text(text) for text in variants]
        encode_seconds = time.perf_counter() - started
        started = time.perf_counter()
        search_batch = getattr(retriever, "search_batch", None)
        values = search_batch(vectors, top_k=pool_size) if callable(search_batch) else [retriever.search(vector, top_k=pool_size) for vector in vectors]
        print(f"[KIS] {source_name} variants={len(variants)} encode={encode_seconds:.2f}s faiss={time.perf_counter()-started:.2f}s pool={pool_size}")
        return values

    @staticmethod
    def _rrf_fuse(ranked_lists: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
        """Fuse query variants inside a single source (kept test-public)."""
        merged: dict[str, dict[str, Any]] = {}
        for variant_index, candidates in enumerate(ranked_lists):
            seen: set[str] = set()
            for rank, candidate in enumerate(candidates, start=1):
                key = candidate_identity(candidate)
                if key in seen:
                    continue
                seen.add(key)
                item = merged.setdefault(key, candidate.copy())
                item.setdefault("query_variants", []).append(variant_index)
                item["rrf_score"] = float(item.get("rrf_score", 0.0)) + 1.0 / (config.KIS_RRF_K + rank)
                item["clip_score"] = max(float(item.get("clip_score", float("-inf"))), float(candidate.get("score", 0.0)))
        result = list(merged.values())
        for item in result:
            item["query_variants"] = sorted(set(item["query_variants"]))
            item["score"] = float(item["rrf_score"])
        return sorted(result, key=lambda item: (-float(item["score"]), -float(item["clip_score"]), candidate_identity(item)))

    def _retrieve_source(self, source_name: str, encoder, retriever, queries: list[str], top_k: int) -> list[dict[str, Any]]:
        variants = self._unique_strings([part for query in queries for part in self._query_variants(query, encoder)])
        return self._rrf_fuse(self._search_variants(encoder, retriever, variants, self._candidate_pool_size(top_k, retriever), source_name))

    def _qwen_analysis(self, english_query: str, *, query_id: str | None) -> dict[str, list[str]]:
        default = {"visual_queries": [], "metadata_queries": [], "ocr_queries": [], "visible_constraints": [], "factual_entities": [], "analysis_ok": []}
        if not self.enable_qwen:
            return default
        evidence = self.evidence_cache.context(query_id) if self.evidence_cache and query_id else ""
        try:
            try:
                raw = self.vlm_pipeline.analyze_kis_query(english_query, evidence_text=evidence)
            except TypeError:
                raw = self.vlm_pipeline.analyze_kis_query(english_query)
            raw = raw or {}
            aliases = {
                "visual_queries": ["visual_queries", "retrieval_queries", "expansions"],
                "metadata_queries": ["metadata_queries", "factual_entities"],
                "ocr_queries": ["ocr_queries"],
                "visible_constraints": ["visible_constraints", "must_have"],
                "factual_entities": ["factual_entities"],
            }
            output = {key: self._unique_strings([entry for alias in names for entry in list(raw.get(alias) or [])]) for key, names in aliases.items()}
            output["analysis_ok"] = ["true"]
            return output
        except Exception as exc:
            if self.strict_sources:
                raise
            print(f"[KIS] Qwen query analysis failed; retaining non-Qwen sources: {exc}")
            return default

    def _retrieve_metadata(self, queries: list[str]) -> list[dict[str, Any]]:
        if self.media_retriever is None:
            return []
        by_video: dict[str, dict[str, Any]] = {}
        for query in queries:
            for rank, result in enumerate(self.media_retriever.search(query, top_k=self.candidate_budget), start=1):
                video_id = self._canonical_video_id(result.get("video_id"))
                if not video_id:
                    continue
                item = by_video.setdefault(video_id, result.copy())
                item["video_id"] = video_id
                item["metadata_rrf_score"] = float(item.get("metadata_rrf_score", 0.0)) + 1.0 / (config.KIS_RRF_K + rank)
                item["metadata_score"] = max(float(item.get("metadata_score", float("-inf"))), float(result.get("metadata_score", 0.0)))
                item["metadata_rank"] = min(int(item.get("metadata_rank", rank)), rank)
                item["score"] = float(item["metadata_rrf_score"])
        return sorted(by_video.values(), key=lambda item: (-float(item["metadata_rrf_score"]), str(item["video_id"])))

    def _local_search(self, encoder, retriever, queries: list[str], video_ids: list[str], source_name: str) -> list[dict[str, Any]]:
        variants = self._unique_strings([part for query in queries for part in self._query_variants(query, encoder)])
        if not variants or not video_ids:
            return []
        encode_batch = getattr(encoder, "encode_text_batch", None)
        vectors = encode_batch(variants) if callable(encode_batch) else [encoder.encode_text(query) for query in variants]
        combined: list[list[dict[str, Any]]] = [[] for _ in variants]
        for video_id in video_ids:
            try:
                search_batch = getattr(retriever, "search_in_video_batch", None)
                per_variant = search_batch(vectors, video_id=video_id, top_k=self.local_frame_budget) if callable(search_batch) else [retriever.search_in_video(vector, video_id, self.local_frame_budget) for vector in vectors]
            except AttributeError:
                per_variant = [[item for item in retriever.search(vector, self._candidate_pool_size(self.local_frame_budget, retriever)) if self._canonical_video_id(item.get("video_id")) == video_id][:self.local_frame_budget] for vector in vectors]
            for index, rows in enumerate(per_variant):
                combined[index].extend(rows)
        values = self._rrf_fuse(combined)
        print(f"[KIS] local {source_name} videos={len(video_ids)} variants={len(variants)} frames={len(values)}")
        return values

    @staticmethod
    def _by_video(candidates: list[dict[str, Any]], video_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
        """Group only materialized frame candidates, never video-only metadata."""
        output = {video_id: [] for video_id in video_ids}
        for candidate in candidates:
            video_id = str(candidate.get("video_id", "")).removesuffix(".mp4")
            if (
                video_id in output
                and candidate.get("frame_id") is not None
                and str(candidate.get("keyframe_name", "")).strip()
            ):
                output[video_id].append(candidate)
        for rows in output.values():
            rows.sort(key=lambda item: (-float(item.get("score", 0.0)), candidate_identity(item)))
        return output

    @staticmethod
    def _qwen_ranked(candidates: list[dict[str, Any]], query: str, constraints: list[str], vlm) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for input_rank, candidate in enumerate(candidates, start=1):
            path = config.keyframe_path(candidate["video_id"], candidate["keyframe_name"])
            try:
                details_fn = getattr(vlm, "score_kis_match_details", None)
                details = details_fn(str(path), query, constraints) if callable(details_fn) else None
                if details is None:
                    score = vlm.score_kis_match(str(path), query, constraints)
                    details = {"score": score, "visible_requirements": []} if score is not None else None
            except Exception as exc:
                print(f"[KIS] Qwen visual scoring skipped for {path}: {exc}")
                continue
            if not details or details.get("score") is None:
                continue
            item = candidate.copy()
            item["qwen_match_score"] = int(details["score"])
            item["qwen_visible_requirements"] = list(details.get("visible_requirements") or [])
            item["qwen_input_rank"] = input_rank
            output.append(item)
        return sorted(output, key=lambda item: (-int(item["qwen_match_score"]), int(item["qwen_input_rank"])))

    def execute(self, query: str, top_k: int = 100, object_labels: str = "", *, query_id: str | None = None) -> list[dict[str, Any]]:
        del object_labels
        english = translate_vi_to_en(query)
        if english != query:
            print(f"Translated query: {english}")
        analysis = self._qwen_analysis(english, query_id=query_id)
        visual = self._unique_strings(analysis["visual_queries"], excluded={english})[:self.query_variant_limit]
        metadata_queries = self._unique_strings([english, *analysis["metadata_queries"], *analysis["factual_entities"]])[:self.query_variant_limit + 1]
        ocr_queries = self._unique_strings([english, *analysis["ocr_queries"]])[:self.query_variant_limit + 1]

        global_sources: list[tuple[str, list[dict[str, Any]]]] = []
        baseline = self._retrieve_source("clip_vitb32_original", self.encoder, self.retriever, [english], top_k)
        if not baseline:
            return []
        global_sources.append(("clip_vitb32_original", baseline))
        if visual:
            values = self._retrieve_source("clip_vitb32_visual", self.encoder, self.retriever, visual, top_k)
            if values:
                global_sources.append(("clip_vitb32_visual", values))
        for name, (secondary_encoder, secondary_retriever) in self.secondary_retrievers.items():
            original = self._retrieve_source(f"{name}_original", secondary_encoder, secondary_retriever, [english], top_k)
            if original:
                global_sources.append((f"{name}_original", original))
            if visual:
                expanded = self._retrieve_source(f"{name}_visual", secondary_encoder, secondary_retriever, visual, top_k)
                if expanded:
                    global_sources.append((f"{name}_visual", expanded))
        media_videos: list[dict[str, Any]] = []
        if self.media_retriever is not None:
            try:
                media_videos = self._retrieve_metadata(metadata_queries)
                if media_videos:
                    global_sources.append(("btc_media_e5", media_videos))
            except Exception as exc:
                if self.strict_sources:
                    raise
                print(f"[KIS] Media E5 retrieval skipped: {exc}")

        videos = fuse_video_rankings(global_sources, rrf_k=config.KIS_RRF_K)[:self.video_budget]
        video_ids = [str(item["video_id"]) for item in videos]
        if not video_ids:
            return []
        local_sources: list[tuple[str, list[dict[str, Any]]]] = []
        sources = [("clip_vitb32", (self.encoder, self.retriever)), *self.secondary_retrievers.items()]
        for source_name, (source_encoder, source_retriever) in sources:
            original = self._local_search(source_encoder, source_retriever, [english], video_ids, f"{source_name}_original")
            if original:
                local_sources.append((f"{source_name}_original", original))
            if visual:
                expanded = self._local_search(source_encoder, source_retriever, visual, video_ids, f"{source_name}_visual")
                if expanded:
                    local_sources.append((f"{source_name}_visual", expanded))
        if not local_sources:
            return []

        if media_videos:
            global_by_video = self._by_video([item for _name, rows in global_sources for item in rows], video_ids)
            media_frames: list[dict[str, Any]] = []
            for item in media_videos:
                for frame in global_by_video.get(str(item["video_id"]), [])[:self.local_frame_budget]:
                    enriched = frame.copy()
                    enriched.update({key: value for key, value in item.items() if key.startswith("metadata_") or key == "metadata_text"})
                    media_frames.append(enriched)
            if media_frames:
                local_sources.append(("btc_media_e5", media_frames))

        fused = fuse_rankings(local_sources, rrf_k=config.KIS_RRF_K)
        if self.ocr is not None:
            try:
                pool = stratified_candidates(self._by_video(fused, video_ids), videos, limit=self.ocr_candidate_budget)
                lists = [self.ocr.rank(text, pool, lambda item: config.keyframe_path(item["video_id"], item["keyframe_name"])) for text in ocr_queries]
                ranked = self._rrf_fuse(lists)
                if ranked:
                    local_sources.append(("candidate_ocr_e5", ranked))
            except Exception as exc:
                if self.strict_sources:
                    raise
                print(f"[KIS] Candidate OCR retrieval skipped: {exc}")

        pre_qwen = fuse_rankings(local_sources, rrf_k=config.KIS_RRF_K)
        qwen_ranked: list[dict[str, Any]] = []
        if self.enable_qwen and analysis["analysis_ok"]:
            started = time.perf_counter()
            pool = stratified_candidates(self._by_video(pre_qwen, video_ids), videos, limit=self.vlm_top_k)
            qwen_ranked = self._qwen_ranked(pool, english, analysis["visible_constraints"], self.vlm_pipeline)
            print(f"[KIS] Qwen scored={len(qwen_ranked)} in {time.perf_counter()-started:.2f}s")
            if qwen_ranked:
                local_sources.append(("qwen_visual_evidence", qwen_ranked))

        final_frames = fuse_rankings(local_sources, rrf_k=config.KIS_RRF_K)
        final_video_sources = list(global_sources)
        if qwen_ranked:
            final_video_sources.append(("qwen_visual_evidence", qwen_ranked))
        final_videos = fuse_video_rankings(final_video_sources, rrf_k=config.KIS_RRF_K)[:self.video_budget]
        final_ids = [str(item["video_id"]) for item in final_videos]
        by_video = self._by_video(final_frames, final_ids)
        if self.frame_neighborhood is not None:
            by_video = {video_id: self.frame_neighborhood.expand_ranked(rows, count=self.neighborhood_count) for video_id, rows in by_video.items()}
        results = d_hondt_allocate(final_videos, by_video, limit=top_k)
        print("[KIS] " + " ".join(f"{name}={len(rows)}" for name, rows in local_sources) + f" videos={len(final_videos)} output={len(results)}")
        return results
