import sys
import json
import re
from pathlib import Path
from deep_translator import GoogleTranslator
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config


def _rrf_fusion(ranked_lists: list, k: int = None) -> list:
    """
    Improvement #7: Reciprocal Rank Fusion.
    Fuses multiple FAISS result lists into one ranked list.
    """
    if k is None:
        k = config.VQA_RRF_K
    scores = {}
    for ranked_list in ranked_lists:
        for rank, candidate in enumerate(ranked_list):
            fid = candidate["faiss_idx"]
            if fid not in scores:
                scores[fid] = {"rrf_score": 0.0, "data": candidate}
            scores[fid]["rrf_score"] += 1.0 / (k + rank + 1)
    fused = sorted(scores.values(), key=lambda x: x["rrf_score"], reverse=True)
    for item in fused:
        item["data"]["score"] = item["rrf_score"]
    return [item["data"] for item in fused]


def _external_search(query: str) -> str:
    """
    Improvement #12: External fact lookup via DuckDuckGo.
    Returns a short text summary from top results.
    """
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if results:
            snippets = [r.get("body", "") for r in results[:2]]
            return " ".join(snippets)[:500]
    except Exception as e:
        print(f"[VQA] External search failed: {e}")
    return ""


class VQATask:
    def __init__(self, encoder, retriever, object_filter=None, vlm_pipeline=None, semantic_filter=None):
        self.encoder = encoder
        self.retriever = retriever
        self.object_filter = object_filter        # Original ObjectFilter (still used for filter_candidates)
        self.vlm_pipeline = vlm_pipeline
        self.semantic_filter = semantic_filter    # SemanticObjectFilter (Improvement #9)

    def execute(self, question: str, top_k: int = None, progress_callback=None) -> tuple:
        if top_k is None:
            top_k = config.VQA_RERANK_K

        # ── Step 1: Translate ─────────────────────────────────────────────
        try:
            english_question = GoogleTranslator(source="vi", target="en").translate(question)
        except Exception as e:
            print(f"[VQA] Translation failed: {e}")
            english_question = question

        # ── Step 2: LLM Query Analysis (Improvements #7, #8, #11, #12) ───
        analysis = {
            "needs_fact_lookup": False,
            "fact_query": "",
            "retrieval_description": english_question,
            "vlm_question": english_question,
            "paraphrases": [],
            "objects_required": [],
        }
        if self.vlm_pipeline:
            try:
                analysis = self.vlm_pipeline.analyze_vqa_query(english_question)
            except Exception as e:
                print(f"[VQA] analyze_vqa_query failed: {e}. Using defaults.")

        retrieval_desc = analysis.get("retrieval_description", english_question)
        vlm_question   = analysis.get("vlm_question", question)
        paraphrases    = analysis.get("paraphrases", [])[:config.VQA_PARAPHRASE_N]
        objects_req    = analysis.get("objects_required", [])

        # ── Step 3: External Fact Lookup (Improvement #12) ────────────────
        if analysis.get("needs_fact_lookup") and analysis.get("fact_query"):
            fact_text = _external_search(analysis["fact_query"])
            if fact_text:
                # Ask LLM to rewrite retrieval_desc with the resolved fact
                if self.vlm_pipeline:
                    try:
                        rewrite_prompt = (
                            f"Rewrite this video search description using the following fact.\n"
                            f"Original: '{retrieval_desc}'\n"
                            f"Fact: '{fact_text[:200]}'\n"
                            f"Output ONLY the rewritten description, no other text."
                        )
                        retrieval_desc = self.vlm_pipeline._text_only_generate(rewrite_prompt, max_new_tokens=80).strip()
                    except Exception as e:
                        print(f"[VQA] Fact rewrite failed: {e}")

        # ── Step 4: Multi-Query FAISS + RRF (Improvements #7, #11) ────────
        candidates_per_k = max(config.VQA_CANDIDATES, top_k * 4)
        all_queries = [retrieval_desc] + paraphrases
        ranked_lists = []
        for q in all_queries:
            try:
                vec = self.encoder.encode_text(q)
                ranked_lists.append(self.retriever.search(query_vector=vec, top_k=candidates_per_k))
            except Exception as e:
                print(f"[VQA] FAISS search failed for query '{q[:40]}': {e}")

        if len(ranked_lists) > 1:
            candidates = _rrf_fusion(ranked_lists)
        elif ranked_lists:
            candidates = ranked_lists[0]
        else:
            return []

        # ── Step 5: Semantic Object Filter (Improvements #3, #9) ──────────
        if self.semantic_filter and objects_req:
            # Semantic filter already extracted objects from LLM, pass directly
            if self.object_filter:
                candidates = self.object_filter.filter_candidates(
                    candidates, objects_req, mode="boost"
                )
        elif self.object_filter and not self.semantic_filter:
            # Fallback: use old Regex extraction if no semantic filter
            all_labels = self.object_filter.get_all_labels()
            query_lower = english_question.lower()
            req_objs = []
            for label in all_labels:
                pattern = r'\b' + re.escape(label.lower()) + r'(?:s|es)?\b'
                if re.search(pattern, query_lower):
                    req_objs.append(label)
            if req_objs:
                candidates = self.object_filter.filter_candidates(candidates, req_objs, mode="boost")

        # Improvement #9: dynamic weight w=0 if no objects required
        # (handled implicitly: if objects_req is empty, no boost is applied above)

        candidates = candidates[:top_k * 2]

        # ── Step 6: VLM Verification + Rerank (Improvement #5) ────────────
        if self.vlm_pipeline:
            for c in candidates:
                try:
                    verified = self.vlm_pipeline.verify_frame(
                        str(config.KEYFRAMES_DIR / c["video_id"] / (c["keyframe_name"] + ".jpg")), retrieval_desc
                    )
                    c["verified"] = verified
                    if verified:
                        c["score"] = c.get("score", 0) + config.VQA_VERIFICATION_BOOST
                except Exception as e:
                    c["verified"] = True  # Keep frame if verification errors
            candidates.sort(key=lambda x: x.get("score", 0), reverse=True)

        candidates = candidates[:top_k]

        # ── Step 7: VLM Answering ──────────────────────────────────────────
        for c in candidates:
            if self.vlm_pipeline:
                try:
                    c["answer"] = self.vlm_pipeline.answer_question(
                        str(config.KEYFRAMES_DIR / c["video_id"] / (c["keyframe_name"] + ".jpg")), vlm_question
                    )
                except Exception as e:
                    c["answer"] = f"[VLM Error: {e}]"
            else:
                c["answer"] = "[Manual Review Required - No VLM loaded]"

        return candidates, analysis



