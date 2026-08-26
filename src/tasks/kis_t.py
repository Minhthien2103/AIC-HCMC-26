"""GPU-oriented textual known-item search.

The KIS path deliberately keeps temporal alignment in TRAKE. KIS first
maximises recall with token-safe CLIP/RRF retrieval, then lets Qwen2-VL promote
only visually verified keyframes while retaining an untouched CLIP fallback.
"""

from __future__ import annotations
from src.tasks.gemini_api import GeminiPipeline
gemini_api = GeminiPipeline()


import re
import time
from typing import Any

from src import config
from src.utils.translation import translate_vi_to_en


_SENTENCE_BOUNDARY_RE = re.compile(r"(?:\n+|(?<=[.!?])\s+)")
_CLAUSE_BOUNDARY_RE = re.compile(r"(?<=[,;:])\s+|(?<=[–—])\s+")


class KIStask:
    """KIS retrieval with deterministic CLIP recall and optional Qwen reranking."""

    def __init__(
        self,
        encoder,
        retriever,
        object_filter=None,
        vlm_pipeline=None,
        *,
        enable_qwen: bool = False,
        vlm_top_k: int | None = None,
    ):
        self.encoder = encoder
        self.retriever = retriever
        # Kept for the Streamlit constructor/API. Automatic OpenImages score
        # boosting is intentionally not applied to RRF-ranked KIS candidates.
        self.object_filter = object_filter
        self.vlm_pipeline = vlm_pipeline
        self.enable_qwen = bool(enable_qwen and vlm_pipeline is not None)
        self.vlm_top_k = int(vlm_top_k or config.KIS_VLM_TOP_K)

    def _context_limit(self) -> int:
        return max(4, int(getattr(self.encoder, "text_context_length", 77)))

    def _token_count(self, text: str) -> int:
        counter = getattr(self.encoder, "text_token_count", None)
        if callable(counter):
            return int(counter(text))
        # Test doubles do not expose a tokenizer. This conservative fallback
        # preserves the same no-truncation property for their whitespace text.
        return len(re.findall(r"\S+", text)) + 2

    def _split_oversized_fragment(self, fragment: str, budget: int) -> list[str]:
        """Split one long sentence while preserving every word in order."""
        if self._token_count(fragment) <= budget:
            return [fragment]

        clauses = [part.strip() for part in _CLAUSE_BOUNDARY_RE.split(fragment) if part.strip()]
        if len(clauses) > 1:
            pieces: list[str] = []
            for clause in clauses:
                pieces.extend(self._split_oversized_fragment(clause, budget))
            return pieces

        words = fragment.split()
        pieces = []
        current: list[str] = []
        for word in words:
            candidate = " ".join([*current, word])
            if current and self._token_count(candidate) > budget:
                pieces.append(" ".join(current))
                current = [word]
            else:
                current.append(word)
        if current:
            pieces.append(" ".join(current))
        return pieces

    def _query_variants(self, query: str) -> list[str]:
        """Return complete, non-truncated CLIP-sized chunks of a query.

        The previous implementation retained the first five clauses only. A
        long KIS description now contributes every constraint, including the
        final sentence, to at least one retrieval variant.
        """
        normalized = re.sub(r"\s+", " ", query).strip()
        if not normalized:
            return []

        budget = self._context_limit()
        if self._token_count(normalized) <= budget:
            return [normalized]

        fragments = [part.strip() for part in _SENTENCE_BOUNDARY_RE.split(normalized) if part.strip()]
        atomic = [
            piece
            for fragment in fragments
            for piece in self._split_oversized_fragment(fragment, budget)
            if piece
        ]
        variants: list[str] = []
        current = ""
        for fragment in atomic:
            candidate = f"{current} {fragment}".strip()
            if current and self._token_count(candidate) > budget:
                variants.append(current)
                current = fragment
            else:
                current = candidate
        if current:
            variants.append(current)
        return list(dict.fromkeys(variants))

    def _candidate_pool_size(self, top_k: int) -> int:
        available = getattr(getattr(self.retriever, "index", None), "ntotal", None)
        pool_size = max(int(top_k), int(config.KIS_CANDIDATES_PER_VARIANT))
        return min(pool_size, int(available)) if available is not None else pool_size

    def _search_variants(self, variants: list[str], pool_size: int) -> list[list[dict[str, Any]]]:
        if not variants:
            return []
        started = time.perf_counter()
        batch_encoder = getattr(self.encoder, "encode_text_batch", None)
        if callable(batch_encoder):
            vectors = batch_encoder(variants)
        else:  # pragma: no cover - compatibility with lightweight test doubles
            vectors = [self.encoder.encode_text(variant) for variant in variants]
        encode_seconds = time.perf_counter() - started

        started = time.perf_counter()
        batch_search = getattr(self.retriever, "search_batch", None)
        if callable(batch_search):
            ranked_lists = batch_search(vectors, top_k=pool_size)
        else:  # pragma: no cover - compatibility with lightweight test doubles
            ranked_lists = [self.retriever.search(vector, top_k=pool_size) for vector in vectors]
        search_seconds = time.perf_counter() - started
        print(
            f"[KIS] CLIP variants={len(variants)} encode={encode_seconds:.2f}s "
            f"faiss={search_seconds:.2f}s pool={pool_size}"
        )
        return ranked_lists

    @staticmethod
    def _rrf_fuse(ranked_lists: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
        by_index: dict[int, dict[str, Any]] = {}
        for variant_index, candidates in enumerate(ranked_lists):
            for rank, candidate in enumerate(candidates):
                faiss_idx = int(candidate["faiss_idx"])
                if faiss_idx not in by_index:
                    merged = candidate.copy()
                    merged["clip_score"] = float(candidate["score"])
                    merged["rrf_score"] = 0.0
                    merged["query_variants"] = []
                    by_index[faiss_idx] = merged
                merged = by_index[faiss_idx]
                merged["rrf_score"] += 1.0 / (config.KIS_RRF_K + rank + 1)
                merged["clip_score"] = max(float(merged["clip_score"]), float(candidate["score"]))
                merged["query_variants"].append(variant_index)

        fused = list(by_index.values())
        for candidate in fused:
            candidate["query_variants"] = sorted(set(candidate["query_variants"]))
            candidate["score"] = float(candidate["rrf_score"])
        fused.sort(key=lambda candidate: (float(candidate["score"]), float(candidate["clip_score"])), reverse=True)
        return fused

    def _retrieve(self, variants: list[str], pool_size: int) -> list[dict[str, Any]]:
        return self._rrf_fuse(self._search_variants(variants, pool_size))

    @staticmethod
    def _unique_strings(values: list[str], excluded: set[str] | None = None) -> list[str]:
        blocked = excluded or set()
        result: list[str] = []
        seen = set(blocked)
        for value in values:
            normalized = re.sub(r"\s+", " ", str(value)).strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                result.append(normalized)
        return result

    def _qwen_analysis(self, english_query: str) -> dict[str, list[str]]:
        default = {"retrieval_queries": [], "must_have": [], "expansions": []}
        if not self.enable_qwen:
            return default
        try:
            analysis = self.vlm_pipeline.analyze_kis_query(english_query)
            return {
                key: self._unique_strings(list(analysis.get(key) or []))
                for key in default
            }
        except Exception as exc:
            print(f"[KIS] Qwen query analysis failed; using CLIP baseline only: {exc}")
            return default

    @staticmethod
    def _select_vlm_candidates(
        baseline: list[dict[str, Any]],
        expanded: list[dict[str, Any]],
        count: int,
    ) -> list[dict[str, Any]]:
        """Reserve half of visual checks for the deterministic baseline."""
        selected: list[dict[str, Any]] = []
        seen: set[int] = set()
        baseline_quota = (count + 1) // 2

        def add(source: list[dict[str, Any]], limit: int | None = None) -> None:
            for candidate in source:
                if limit is not None and len(selected) >= limit:
                    return
                faiss_idx = int(candidate["faiss_idx"])
                if faiss_idx not in seen:
                    seen.add(faiss_idx)
                    selected.append(candidate.copy())
                if len(selected) >= count:
                    return

        add(baseline, baseline_quota)
        add(expanded)
        add(baseline)
        return selected[:count]

    def _qwen_rerank(
        self,
        candidates: list[dict[str, Any]],
        original_query: str,
        must_have: list[str],
    ) -> list[dict[str, Any]]:
        if not self.enable_qwen:
            return []
        started = time.perf_counter()
        scored: list[dict[str, Any]] = []
        for candidate in candidates:
            image_path = config.keyframe_path(candidate["video_id"], candidate["keyframe_name"])
            score = self.vlm_pipeline.score_kis_match(str(image_path), original_query, must_have)
            if score is not None:
                result = candidate.copy()
                result["qwen_match_score"] = int(score)
                scored.append(result)
        print(f"[KIS] Qwen visual rerank={len(candidates)} images in {time.perf_counter() - started:.2f}s")
        return scored

    @staticmethod
    def _ordered_unique(
        groups: list[list[dict[str, Any]]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        seen: set[int] = set()
        for group in groups:
            for candidate in group:
                faiss_idx = int(candidate["faiss_idx"])
                if faiss_idx not in seen:
                    seen.add(faiss_idx)
                    output.append(candidate)
                if len(output) >= top_k:
                    return output
        return output


    # -- Multi-scene temporal keywords (Vietnamese) --
    _MULTI_SCENE_MARKERS = re.compile(
        r"(?:bat dau|ket thuc|sau do|tiep theo|truoc do|tiep den"
        r"|bắt đầu|kết thúc|sau đó|tiếp theo|trước đó|tiếp đến"
        r"|rồi|sau vài giây|ngay sau)",
        re.IGNORECASE,
    )

    def _is_multi_scene(self, query: str) -> bool:
        """Detect if a KIS query describes multiple temporal scenes."""
        return len(self._MULTI_SCENE_MARKERS.findall(query)) >= 1

    def _split_scenes(self, query: str) -> list[str]:
        """Split a multi-scene query into individual scene descriptions."""
        # Split on sentence boundaries
        scenes = [s.strip() for s in _SENTENCE_BOUNDARY_RE.split(query) if s.strip()]
        # If we only got 1 chunk, the query isn't really multi-scene
        return scenes if len(scenes) > 1 else [query]

    def _video_level_rrf(
        self,
        scenes: list[str],
        english_query: str,
        pool_size: int,
    ) -> list[dict[str, Any]]:
        video_rrf: dict[str, float] = {}
        all_candidates: dict[str, dict[int, dict[str, Any]]] = {}  # video_id -> faiss_idx -> cand
        
        for scene_idx, scene in enumerate(scenes):
            scene_en = translate_vi_to_en(scene)
            variants = self._query_variants(scene_en)
            if not variants:
                continue
            scene_results = self._retrieve(variants, pool_size)
            
            scene_video_scores: dict[str, float] = {}
            for cand in scene_results:
                vid = cand["video_id"]
                score = float(cand.get("score", 0.0))
                scene_video_scores[vid] = max(scene_video_scores.get(vid, float("-inf")), score)
                
                if vid not in all_candidates:
                    all_candidates[vid] = {}
                idx = int(cand["faiss_idx"])
                if idx not in all_candidates[vid]:
                    all_candidates[vid][idx] = cand.copy()
                    all_candidates[vid][idx]["scene_idx"] = scene_idx
                else:
                    all_candidates[vid][idx]["scene_idx"] = max(all_candidates[vid][idx]["scene_idx"], scene_idx)
                    all_candidates[vid][idx]["score"] = max(float(all_candidates[vid][idx]["score"]), score)
            
            sorted_vids = sorted(scene_video_scores.keys(), key=lambda v: scene_video_scores[v], reverse=True)
            for rank, vid in enumerate(sorted_vids):
                video_rrf[vid] = video_rrf.get(vid, 0.0) + 1.0 / (config.KIS_RRF_K + rank + 1)
        
        top_videos = sorted(video_rrf.keys(), key=lambda v: video_rrf[v], reverse=True)[:20]
        print(f"[KIS] Multi-scene: {len(scenes)} scenes, top videos: {top_videos[:5]}")
        
        output: list[dict[str, Any]] = []
        for vid in top_videos:
            vid_cands = list(all_candidates.get(vid, {}).values())
            # Boost frames that come from the LAST scene (often the target keyframe)
            # Add +100 to score to force them to the top of this video's candidates
            for c in vid_cands:
                c["score"] = float(c["score"])
                if c["scene_idx"] == len(scenes) - 1:
                    c["score"] += 100.0
            
            vid_cands.sort(key=lambda c: c["score"], reverse=True)
            for cand in vid_cands:
                boosted = cand.copy()
                # Preserve video-level order
                boosted["score"] = cand["score"] + (video_rrf[vid] * 1000)
                output.append(boosted)
        
        output.sort(key=lambda c: c["score"], reverse=True)
        return output

    def execute(self, query: str, top_k: int = 100, object_labels: str = "") -> list[dict[str, Any]]:
        english_query = translate_vi_to_en(query)
        if english_query != query:
            print(f"Translated query: {english_query}")

        pool_size = self._candidate_pool_size(top_k)
        
        # Detect multi-scene queries and use video-level RRF
        is_multi = self._is_multi_scene(query)
        scenes = self._split_scenes(query) if is_multi else []
        
        if is_multi and len(scenes) > 1:
            print(f"[KIS] Multi-scene query detected ({len(scenes)} scenes)")
            video_boosted = self._video_level_rrf(scenes, english_query, pool_size)
        else:
            video_boosted = []
        
        baseline_variants = self._query_variants(english_query)
        baseline = self._retrieve(baseline_variants, pool_size)
        if not baseline and not video_boosted:
            return []
        
        # Merge video-level RRF results into baseline (video-boosted first)
        if video_boosted:
            seen_idx = set()
            merged = []
            for cand in video_boosted:
                idx = int(cand["faiss_idx"])
                if idx not in seen_idx:
                    seen_idx.add(idx)
                    merged.append(cand)
            for cand in baseline:
                idx = int(cand["faiss_idx"])
                if idx not in seen_idx:
                    seen_idx.add(idx)
                    merged.append(cand)
            baseline = merged

        analysis = self._qwen_analysis(english_query)
        qwen_hints = self._unique_strings(
            [*analysis["retrieval_queries"], *analysis["expansions"]],
            excluded=set(baseline_variants),
        )
        expanded_variants = [*baseline_variants]
        for hint in qwen_hints:
            expanded_variants.extend(self._query_variants(hint))
        expanded_variants = self._unique_strings(expanded_variants)
        expanded = (
            self._retrieve(expanded_variants, pool_size)
            if len(expanded_variants) > len(baseline_variants)
            else baseline
        )

        manual_objects = self._unique_strings(object_labels.split(",")) if object_labels else []
        must_have = self._unique_strings([*analysis["must_have"], *manual_objects])
        
        # Apply strict object pre-filtering via parquet DB so VLM focuses only on semantic context
        if self.object_filter and must_have:
            baseline = self.object_filter.filter_candidates(baseline, must_have, mode="boost")
            if expanded:
                expanded = self.object_filter.filter_candidates(expanded, must_have, mode="boost")
                
        # Phase 2: Extreme API Batching (Collage Strategy)
        # Extract Top 10 unique videos from baseline
        top_videos_data = {}
        for cand in baseline:
            vid = cand["video_id"]
            if vid not in top_videos_data:
                top_videos_data[vid] = []
            if len(top_videos_data[vid]) < 3: # Max 3 frames per video for storyboard
                img_path = str(config.keyframe_path(vid, cand["keyframe_name"]))
                top_videos_data[vid].append(img_path)
            if len(top_videos_data) >= 10: # Top 10 videos max
                break
                
        candidate_payload = [{"video_id": vid, "frames": frames} for vid, frames in top_videos_data.items()]
        
        print(f"[Gemini] Sending {len(candidate_payload)} candidate videos for 1-Call tournament...")
        winner_vid = gemini_api.evaluate_kis_candidates(query, candidate_payload)
        
        promoted = []
        if winner_vid and winner_vid != "NONE":
            print(f"[Gemini] Winner selected: {winner_vid}")
            # Promote all frames from the winning video to the top of the list!
            for cand in baseline + expanded:
                if cand["video_id"] == winner_vid:
                    promoted.append(cand.copy())
            
        results = self._ordered_unique([promoted, baseline, expanded], top_k)
        print(
            f"[KIS] baseline={len(baseline)} expanded={len(expanded)} "
            f"qwen_promoted={len(promoted)} output={len(results)}"
        )
        return results
