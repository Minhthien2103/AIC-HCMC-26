"""GPU-oriented textual known-item search.

The KIS path deliberately keeps temporal alignment in TRAKE. KIS first
maximises recall with token-safe CLIP/RRF retrieval, then lets Qwen2-VL promote
only visually verified keyframes while retaining an untouched CLIP fallback.
"""

from __future__ import annotations

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

    def execute(self, query: str, top_k: int = 100, object_labels: str = "") -> list[dict[str, Any]]:
        english_query = translate_vi_to_en(query)
        if english_query != query:
            print(f"Translated query: {english_query}")

        pool_size = self._candidate_pool_size(top_k)
        baseline_variants = self._query_variants(english_query)
        baseline = self._retrieve(baseline_variants, pool_size)
        if not baseline:
            return []

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
        qwen_candidates = self._select_vlm_candidates(baseline, expanded, min(top_k, self.vlm_top_k))
        qwen_scores = self._qwen_rerank(qwen_candidates, english_query, must_have)
        promoted = [
            candidate for candidate in qwen_scores
            if int(candidate["qwen_match_score"]) >= config.KIS_VLM_PROMOTE_MIN_SCORE
        ]
        promoted.sort(
            key=lambda candidate: (int(candidate["qwen_match_score"]), float(candidate["score"])),
            reverse=True,
        )
        results = self._ordered_unique([promoted, baseline, expanded], top_k)
        print(
            f"[KIS] baseline={len(baseline)} expanded={len(expanded)} "
            f"qwen_promoted={len(promoted)} output={len(results)}"
        )
        return results
