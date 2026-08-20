import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.online_pipeline.temporal_search import TemporalSearchEngine
from src.utils.translation import translate_vi_to_en

# RRF constant (shared with VQA)
_RRF_K = 60


def _rrf_fusion(ranked_lists: list, k: int = _RRF_K) -> list:
    """Fuse multiple ranked FAISS result lists with Reciprocal Rank Fusion."""
    scores: dict[int, dict] = {}
    for ranked_list in ranked_lists:
        for rank, candidate in enumerate(ranked_list):
            fid = candidate["faiss_idx"]
            if fid not in scores:
                scores[fid] = {"rrf_score": 0.0, "data": candidate.copy()}
            scores[fid]["rrf_score"] += 1.0 / (k + rank + 1)
    fused = sorted(scores.values(), key=lambda x: x["rrf_score"], reverse=True)
    result = []
    for entry in fused:
        c = entry["data"]
        c["score"] = entry["rrf_score"]
        result.append(c)
    return result


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on Vietnamese/English sentence boundaries."""
    # Split on newlines first, then on . ! ? followed by whitespace/capital
    parts = re.split(r'\n+', text.strip())
    sentences = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Further split on sentence-ending punctuation
        subs = re.split(r'(?<=[.!?])\s+(?=[A-ZAĂÂĐÊÔƠƯ])', part)
        sentences.extend([s.strip() for s in subs if len(s.strip()) > 15])
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for s in sentences:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique if unique else [text.strip()]


class KIStask:
    def __init__(self, encoder, retriever, object_filter=None, vlm_pipeline=None):
        self.encoder = encoder
        self.retriever = retriever
        self.object_filter = object_filter
        self.vlm_pipeline = vlm_pipeline
        self.temporal_engine = TemporalSearchEngine(encoder, retriever, object_filter)

    def _analyze_query(self, english_query: str) -> dict:
        """Use VLM to extract key_scene, camera info, and whether fact lookup is needed.
        Falls back gracefully if VLM is not loaded or fails.
        """
        if self.vlm_pipeline is None:
            return {}
        try:
            prompt = (
                "Analyze this video search query and output ONLY valid JSON with these keys:\n"
                "  key_scene: The single most visually distinctive moment to find in ONE keyframe. "
                "Be concise (<15 words). Include camera angle if mentioned (aerial, close-up, etc.).\n"
                "  needs_fact_lookup: true if the query references a known entity, film, person, or "
                "fact that must be looked up (e.g. a 1975 Spielberg film, a university city). false otherwise.\n"
                "  fact_query: if needs_fact_lookup is true, the short English search query to resolve the fact.\n"
                "  objects_required: list of key physical objects that must be visible.\n"
                f"Query: {english_query}"
            )
            raw = self.vlm_pipeline._text_only_generate(prompt, max_new_tokens=200)
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                import json
                parsed = json.loads(match.group(0))
                return parsed if isinstance(parsed, dict) else {}
        except Exception as e:
            print(f"[KIS] analyze_query failed: {e}")
        return {}

    def _external_search(self, fact_query: str) -> str:
        """Resolve a factual query via Wikipedia summary (no API key needed)."""
        try:
            import urllib.request
            import urllib.parse
            import json as _json
            url = (
                "https://en.wikipedia.org/api/rest_v1/page/summary/"
                + urllib.parse.quote(fact_query.replace(" ", "_"))
            )
            req = urllib.request.Request(url, headers={"User-Agent": "AIC2026/1.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = _json.loads(resp.read())
                return data.get("extract", "")[:300]
        except Exception:
            return ""

    def execute(self, query: str, top_k: int = 100, object_labels: str = "") -> list[dict]:
        # ── Step 1: Translate ───────────────────────────────────────────────
        english_query = translate_vi_to_en(query)
        if english_query != query:
            print(f"[KIS] Translated: {english_query[:80]}")

        # ── Step 2: VLM-based query analysis (#16, #17, #18) ───────────────
        analysis = self._analyze_query(english_query)
        key_scene = str(analysis.get("key_scene") or "").strip()
        objects_req_from_llm = [str(o) for o in (analysis.get("objects_required") or []) if str(o).strip()]

        # ── Step 3: Fact lookup for knowledge-grounded queries (#16) ────────
        if analysis.get("needs_fact_lookup") and analysis.get("fact_query"):
            fact = self._external_search(str(analysis["fact_query"]))
            if fact:
                english_query = f"{english_query}. Context: {fact[:150]}"
                print(f"[KIS] Fact resolved: {fact[:80]}")

        # ── Step 4: Build query list for multi-query RRF (#14, #17, #18) ───
        # Split long descriptions into sentences (fixes 77-token CLIP limit)
        sentences = _split_sentences(english_query)
        all_queries = list(dict.fromkeys(
            ([key_scene] if key_scene else []) + sentences
        ))
        # Cap to avoid too many FAISS calls
        all_queries = all_queries[:6]
        print(f"[KIS] Searching {len(all_queries)} query variants")

        # ── Step 5: FAISS search per query + RRF fusion ─────────────────────
        candidates_per_q = max(top_k * 2, 200)
        ranked_lists = []

        for q in all_queries:
            try:
                vec = self.encoder.encode_text(q)
                if self.temporal_engine.has_temporal_keywords(q):
                    res = self.temporal_engine.execute(q, top_k=candidates_per_q)
                else:
                    res = self.retriever.search(query_vector=vec, top_k=candidates_per_q)
                if res:
                    ranked_lists.append(res)
            except Exception as e:
                print(f"[KIS] FAISS search failed for '{q[:40]}': {e}")

        if not ranked_lists:
            return []

        candidates = _rrf_fusion(ranked_lists) if len(ranked_lists) > 1 else ranked_lists[0]

        # ── Step 6: Object filter boost ────────────────────────────────────
        req_objs = list(objects_req_from_llm)
        if object_labels:
            req_objs += [o.strip() for o in object_labels.split(",") if o.strip()]
        if not req_objs and self.object_filter:
            all_labels = self.object_filter.get_all_labels()
            query_lower = english_query.lower()
            for label in all_labels:
                pattern = r'\b' + re.escape(label.lower()) + r'(?:s|es)?\b'
                if re.search(pattern, query_lower):
                    req_objs.append(label)

        if req_objs and self.object_filter:
            candidates = self.object_filter.filter_candidates(candidates, req_objs, mode="boost")

        return candidates[:top_k]