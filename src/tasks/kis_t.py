"""Known-item search with equal-weight, auditable multi-source retrieval."""

from __future__ import annotations

import re
import time
from typing import Any

from src import config
from src.online_pipeline.rank_fusion import candidate_identity, fuse_rankings
from src.utils.translation import translate_vi_to_en


_SENTENCE_BOUNDARY_RE = re.compile(r"(?:\n+|(?<=[.!?])\s+)")
_CLAUSE_BOUNDARY_RE = re.compile(r"(?<=[,;:])\s+|(?<=[–—])\s+")


class KIStask:
    """Retrieve KIS candidates from CLIP, text metadata, OCR and Qwen evidence.

    All retrievers are ranked independently and fused through equal-weight RRF.
    No source can force a candidate to the top through a hidden threshold.
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
        strict_sources: bool = False,
    ):
        self.encoder = encoder
        self.retriever = retriever
        # Kept only for backwards-compatible constructors. Object label boost
        # is deliberately not part of KIS ranking: it overfits common labels.
        self.object_filter = object_filter
        self.vlm_pipeline = vlm_pipeline
        self.enable_qwen = bool(enable_qwen and vlm_pipeline is not None)
        self.vlm_top_k = int(vlm_top_k or config.KIS_QWEN_RERANK_TOP_K)
        self.secondary_retrievers = dict(secondary_retrievers or {})
        self.media_retriever = media_retriever
        self.ocr = ocr
        self.evidence_cache = evidence_cache
        self.candidate_budget = int(candidate_budget or config.KIS_DUAL_CANDIDATE_BUDGET)
        self.ocr_candidate_budget = int(ocr_candidate_budget or config.KIS_OCR_CANDIDATE_BUDGET)
        self.text_frames_per_video = int(text_frames_per_video or config.KIS_TEXT_FRAMES_PER_VIDEO)
        self.strict_sources = strict_sources

    def _context_limit(self, encoder=None) -> int:
        active = encoder or self.encoder
        return max(4, int(getattr(active, "text_context_length", 77)))

    def _token_count(self, text: str, encoder=None) -> int:
        active = encoder or self.encoder
        counter = getattr(active, "text_token_count", None)
        if callable(counter):
            return int(counter(text))
        return len(re.findall(r"\S+", text)) + 2

    def _split_oversized_fragment(self, fragment: str, budget: int, encoder=None) -> list[str]:
        if self._token_count(fragment, encoder) <= budget:
            return [fragment]
        clauses = [part.strip() for part in _CLAUSE_BOUNDARY_RE.split(fragment) if part.strip()]
        if len(clauses) > 1:
            return [piece for clause in clauses for piece in self._split_oversized_fragment(clause, budget, encoder)]
        words = fragment.split()
        pieces: list[str] = []
        current: list[str] = []
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
        """Split long text safely; every supplied constraint remains searchable."""
        normalized = re.sub(r"\s+", " ", query).strip()
        if not normalized:
            return []
        budget = self._context_limit(encoder)
        if self._token_count(normalized, encoder) <= budget:
            return [normalized]
        fragments = [part.strip() for part in _SENTENCE_BOUNDARY_RE.split(normalized) if part.strip()]
        atomic = [
            piece
            for fragment in fragments
            for piece in self._split_oversized_fragment(fragment, budget, encoder)
            if piece
        ]
        variants: list[str] = []
        current = ""
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

    def _candidate_pool_size(self, top_k: int, retriever=None) -> int:
        active = retriever or self.retriever
        available = getattr(getattr(active, "index", None), "ntotal", None)
        pool_size = max(int(top_k), self.candidate_budget)
        return min(pool_size, int(available)) if available is not None else pool_size

    def _search_variants(self, encoder, retriever, variants: list[str], pool_size: int, source_name: str) -> list[list[dict[str, Any]]]:
        if not variants:
            return []
        started = time.perf_counter()
        batch_encoder = getattr(encoder, "encode_text_batch", None)
        vectors = batch_encoder(variants) if callable(batch_encoder) else [encoder.encode_text(item) for item in variants]
        encode_seconds = time.perf_counter() - started
        started = time.perf_counter()
        batch_search = getattr(retriever, "search_batch", None)
        ranked_lists = batch_search(vectors, top_k=pool_size) if callable(batch_search) else [
            retriever.search(vector, top_k=pool_size) for vector in vectors
        ]
        print(
            f"[KIS] {source_name} variants={len(variants)} encode={encode_seconds:.2f}s "
            f"faiss={time.perf_counter() - started:.2f}s pool={pool_size}"
        )
        return ranked_lists

    @staticmethod
    def _rrf_fuse(ranked_lists: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
        """Fuse variants within one source (kept public for regression tests)."""
        by_identity: dict[str, dict[str, Any]] = {}
        for variant_index, candidates in enumerate(ranked_lists):
            seen_variant: set[str] = set()
            for rank, candidate in enumerate(candidates, start=1):
                key = candidate_identity(candidate)
                if key in seen_variant:
                    continue
                seen_variant.add(key)
                item = by_identity.setdefault(key, candidate.copy())
                item.setdefault("query_variants", []).append(variant_index)
                item["rrf_score"] = float(item.get("rrf_score", 0.0)) + 1.0 / (config.KIS_RRF_K + rank)
                item["clip_score"] = max(float(item.get("clip_score", float("-inf"))), float(candidate.get("score", 0.0)))
        result = list(by_identity.values())
        for item in result:
            item["query_variants"] = sorted(set(item["query_variants"]))
            item["score"] = float(item["rrf_score"])
        result.sort(key=lambda item: (-float(item["score"]), -float(item["clip_score"]), candidate_identity(item)))
        return result

    def _retrieve_source(self, source_name: str, encoder, retriever, query_variants: list[str], top_k: int) -> list[dict[str, Any]]:
        source_variants: list[str] = []
        for query in query_variants:
            source_variants.extend(self._query_variants(query, encoder))
        source_variants = self._unique_strings(source_variants)
        ranked_lists = self._search_variants(
            encoder,
            retriever,
            source_variants,
            self._candidate_pool_size(top_k, retriever),
            source_name,
        )
        return self._rrf_fuse(ranked_lists)

    @staticmethod
    def _unique_strings(values: list[str], excluded: set[str] | None = None) -> list[str]:
        blocked = excluded or set()
        output: list[str] = []
        seen = set(blocked)
        for value in values:
            normalized = re.sub(r"\s+", " ", str(value)).strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                output.append(normalized)
        return output

    @staticmethod
    def _canonical_video_id(value: object) -> str:
        return str(value).strip().removesuffix(".mp4")

    def _qwen_analysis(self, english_query: str, *, query_id: str | None) -> dict[str, list[str]]:
        default = {"retrieval_queries": [], "must_have": [], "expansions": [], "factual_entities": []}
        if not self.enable_qwen:
            return default
        evidence_text = self.evidence_cache.context(query_id) if self.evidence_cache and query_id else ""
        try:
            try:
                analysis = self.vlm_pipeline.analyze_kis_query(english_query, evidence_text=evidence_text)
            except TypeError:  # Existing test/UI adapters with the old signature.
                analysis = self.vlm_pipeline.analyze_kis_query(english_query)
            return {key: self._unique_strings(list(analysis.get(key) or [])) for key in default}
        except Exception as exc:
            if self.strict_sources:
                raise
            print(f"[KIS] Qwen query analysis failed; retaining non-Qwen sources: {exc}")
            return default

    def _metadata_frame_candidates(
        self,
        metadata_matches: list[dict[str, Any]],
        visual_candidates: list[dict[str, Any]],
        original_vector,
    ) -> list[dict[str, Any]]:
        """Turn video-level metadata ranks into visual keyframe candidates."""
        by_video: dict[str, list[dict[str, Any]]] = {}
        for candidate in visual_candidates:
            by_video.setdefault(self._canonical_video_id(candidate.get("video_id")), []).append(candidate)
        for values in by_video.values():
            values.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)

        selected: list[dict[str, Any]] = []
        seen: set[str] = set()
        for metadata in metadata_matches:
            video_id = self._canonical_video_id(metadata.get("video_id"))
            frames = list(by_video.get(video_id, []))
            if len(frames) < self.text_frames_per_video and original_vector is not None:
                try:
                    frames.extend(self.retriever.search_in_video(original_vector, video_id, self.text_frames_per_video))
                except Exception as exc:
                    print(f"[KIS] Could not expand metadata video {video_id}: {exc}")
            count = 0
            for frame in frames:
                key = candidate_identity(frame)
                if key in seen:
                    continue
                seen.add(key)
                item = frame.copy()
                item["metadata_score"] = metadata["metadata_score"]
                item["metadata_rank"] = metadata["metadata_rank"]
                item["metadata_text"] = metadata["metadata_text"]
                selected.append(item)
                count += 1
                if count >= self.text_frames_per_video:
                    break
        return selected

    def _retrieve_metadata(self, queries: list[str]) -> list[dict[str, Any]]:
        """Fuse video-level E5 searches over original and Qwen-derived text."""
        by_video: dict[str, dict[str, Any]] = {}
        for query in queries:
            for rank, result in enumerate(self.media_retriever.search(query, top_k=self.candidate_budget), start=1):
                video_id = self._canonical_video_id(result.get("video_id"))
                if not video_id:
                    continue
                item = by_video.setdefault(video_id, result.copy())
                item["metadata_rrf_score"] = float(item.get("metadata_rrf_score", 0.0)) + 1.0 / (config.KIS_RRF_K + rank)
                item["metadata_score"] = max(float(item.get("metadata_score", float("-inf"))), float(result["metadata_score"]))
                item["metadata_rank"] = min(int(item.get("metadata_rank", rank)), rank)
        output = list(by_video.values())
        output.sort(key=lambda item: (-float(item["metadata_rrf_score"]), -float(item["metadata_score"])))
        return output

    @staticmethod
    def _qwen_ranked(candidates: list[dict[str, Any]], original_query: str, must_have: list[str], vlm_pipeline) -> list[dict[str, Any]]:
        scored: list[dict[str, Any]] = []
        for original_rank, candidate in enumerate(candidates, start=1):
            image_path = config.keyframe_path(candidate["video_id"], candidate["keyframe_name"])
            try:
                details_fn = getattr(vlm_pipeline, "score_kis_match_details", None)
                details = details_fn(str(image_path), original_query, must_have) if callable(details_fn) else None
                if details is None:
                    score = vlm_pipeline.score_kis_match(str(image_path), original_query, must_have)
                    details = {"score": score, "visible_requirements": []} if score is not None else None
            except Exception as exc:
                print(f"[KIS] Qwen visual scoring skipped for {image_path}: {exc}")
                continue
            if not details or details.get("score") is None:
                continue
            item = candidate.copy()
            item["qwen_match_score"] = int(details["score"])
            item["qwen_visible_requirements"] = list(details.get("visible_requirements") or [])
            item["qwen_input_rank"] = original_rank
            scored.append(item)
        # The score only orders Qwen's own evidence list. Its contribution to
        # final rank is still exactly the same RRF vote as every other source.
        scored.sort(key=lambda item: (-int(item["qwen_match_score"]), int(item["qwen_input_rank"])))
        return scored

    def execute(
        self,
        query: str,
        top_k: int = 100,
        object_labels: str = "",
        *,
        query_id: str | None = None,
    ) -> list[dict[str, Any]]:
        del object_labels  # Manual object boosts are intentionally retired for KIS.
        english_query = translate_vi_to_en(query)
        if english_query != query:
            print(f"Translated query: {english_query}")

        analysis = self._qwen_analysis(english_query, query_id=query_id)
        derived_queries = self._unique_strings(
            [*analysis["retrieval_queries"], *analysis["expansions"], *analysis["factual_entities"]],
            excluded={english_query},
        )
        source_queries = [english_query, *derived_queries]
        source_rankings: list[tuple[str, list[dict[str, Any]]]] = []
        # The original description remains a standalone source. Qwen's
        # paraphrases may add recall, but cannot replace it or inherit a
        # hidden multiplier by being mixed into the same ranked list.
        baseline = self._retrieve_source("clip_vitb32", self.encoder, self.retriever, [english_query], top_k)
        if not baseline:
            return []
        source_rankings.append(("clip_vitb32", baseline))
        if derived_queries:
            expanded = self._retrieve_source("clip_vitb32_qwen_queries", self.encoder, self.retriever, derived_queries, top_k)
            if expanded:
                source_rankings.append(("clip_vitb32_qwen_queries", expanded))
        for source_name, (encoder, retriever) in self.secondary_retrievers.items():
            result = self._retrieve_source(source_name, encoder, retriever, [english_query], top_k)
            if result:
                source_rankings.append((source_name, result))
            if derived_queries:
                expanded = self._retrieve_source(f"{source_name}_qwen_queries", encoder, retriever, derived_queries, top_k)
                if expanded:
                    source_rankings.append((f"{source_name}_qwen_queries", expanded))

        # Metadata ranks videos, then CLIP chooses the strongest supplied
        # keyframes in each video. It is a separate RRF source, not a boost.
        if self.media_retriever is not None:
            try:
                original_vector = self.encoder.encode_text(english_query)
                metadata = self._retrieve_metadata(source_queries)
                visual_union = [item for _name, values in source_rankings for item in values]
                media_frames = self._metadata_frame_candidates(metadata, visual_union, original_vector)
                if media_frames:
                    source_rankings.append(("btc_media_e5", media_frames))
            except Exception as exc:
                if self.strict_sources:
                    raise
                print(f"[KIS] Media metadata retrieval skipped: {exc}")

        fused_without_ocr = fuse_rankings(source_rankings, rrf_k=config.KIS_RRF_K)
        if self.ocr is not None:
            try:
                ocr_ranked = self.ocr.rank(
                    english_query,
                    fused_without_ocr[: self.ocr_candidate_budget],
                    lambda item: config.keyframe_path(item["video_id"], item["keyframe_name"]),
                )
                if ocr_ranked:
                    source_rankings.append(("candidate_ocr_e5", ocr_ranked))
            except Exception as exc:
                if self.strict_sources:
                    raise
                print(f"[KIS] Candidate OCR retrieval skipped: {exc}")

        pre_qwen = fuse_rankings(source_rankings, rrf_k=config.KIS_RRF_K)
        if self.enable_qwen:
            started = time.perf_counter()
            qwen_ranked = self._qwen_ranked(
                pre_qwen[: self.vlm_top_k], english_query, analysis["must_have"], self.vlm_pipeline
            )
            print(f"[KIS] Qwen scored={len(qwen_ranked)} in {time.perf_counter() - started:.2f}s")
            if qwen_ranked:
                source_rankings.append(("qwen_visual_evidence", qwen_ranked))

        results = fuse_rankings(source_rankings, rrf_k=config.KIS_RRF_K)[:top_k]
        print("[KIS] " + " ".join(f"{name}={len(rows)}" for name, rows in source_rankings) + f" output={len(results)}")
        return results
