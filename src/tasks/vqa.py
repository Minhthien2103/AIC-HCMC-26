import sys
import re
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.utils.translation import translate_vi_to_en


def _rrf_fusion(ranked_lists: list, k: int = None) -> list:
    """Fuse ranked CLIP lists without changing candidate identity."""
    if k is None:
        k = config.VQA_RRF_K
    scores = {}
    for ranked_list in ranked_lists:
        for rank, candidate in enumerate(ranked_list):
            fid = candidate["faiss_idx"]
            if fid not in scores:
                scores[fid] = {"rrf_score": 0.0, "data": candidate.copy()}
            scores[fid]["rrf_score"] += 1.0 / (k + rank + 1)
    fused = sorted(scores.values(), key=lambda item: item["rrf_score"], reverse=True)
    for item in fused:
        item["data"]["score"] = item["rrf_score"]
    return [item["data"] for item in fused]


def _external_search(query: str) -> str:
    """Best-effort optional lookup; never required for a reproducible run."""
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        return " ".join(result.get("body", "") for result in results[:2])[:500]
    except Exception as exc:
        print(f"[VQA] External search unavailable: {exc}")
        return ""


class VQATask:
    def __init__(self, encoder, retriever, object_filter=None, vlm_pipeline=None, semantic_filter=None):
        self.encoder = encoder
        self.retriever = retriever
        self.object_filter = object_filter
        self.vlm_pipeline = vlm_pipeline
        self.semantic_filter = semantic_filter

    def execute(
        self,
        question: str,
        top_k: int = None,
        progress_callback=None,
        allow_external_search: bool | None = None,
    ) -> tuple[list[dict], dict]:
        top_k = config.VQA_RERANK_K if top_k is None else max(1, min(int(top_k), config.VQA_MAX_CANDIDATES))
        if allow_external_search is None:
            allow_external_search = config.VQA_ENABLE_EXTERNAL_SEARCH

        english_question = translate_vi_to_en(question)

        analysis = {
            "needs_fact_lookup": False,
            "fact_query": "",
            "retrieval_description": english_question,
            "vlm_question": question,
            "paraphrases": [],
            "objects_required": [],
        }
        if self.vlm_pipeline:
            try:
                analysis = self.vlm_pipeline.analyze_vqa_query(english_question)
            except Exception as exc:
                print(f"[VQA] analyze_vqa_query failed: {exc}. Using defaults.")

        retrieval_desc = str(analysis.get("retrieval_description") or english_question)
        vlm_question = str(analysis.get("vlm_question") or question)
        paraphrases = [str(value) for value in (analysis.get("paraphrases") or []) if str(value).strip()]
        paraphrases = paraphrases[: config.VQA_PARAPHRASE_N]
        objects_req = [str(value) for value in (analysis.get("objects_required") or []) if str(value).strip()]

        if allow_external_search and analysis.get("needs_fact_lookup") and analysis.get("fact_query"):
            fact_text = _external_search(str(analysis["fact_query"]))
            if fact_text and self.vlm_pipeline:
                try:
                    retrieval_desc = self.vlm_pipeline._text_only_generate(
                        f"Rewrite this video search description using this fact.\nOriginal: {retrieval_desc}\nFact: {fact_text}\nOutput only the rewritten description.",
                        max_new_tokens=80,
                    ).strip()
                except Exception as exc:
                    print(f"[VQA] Fact rewrite failed: {exc}")

        candidates_per_k = min(config.VQA_MAX_CANDIDATES, max(config.VQA_CANDIDATES, top_k * 2))
        ranked_lists = []
        for retrieval_query in [retrieval_desc, *paraphrases]:
            try:
                vector = self.encoder.encode_text(retrieval_query)
                ranked_lists.append(self.retriever.search(vector, top_k=candidates_per_k))
            except Exception as exc:
                print(f"[VQA] FAISS search failed: {exc}")

        if not ranked_lists:
            return [], analysis
        candidates = _rrf_fusion(ranked_lists) if len(ranked_lists) > 1 else ranked_lists[0]

        if self.semantic_filter:
            try:
                objects_req = list(dict.fromkeys([*objects_req, *self.semantic_filter.extract_objects(english_question)]))
            except Exception as exc:
                print(f"[VQA] Semantic object filter unavailable: {exc}")
        if self.object_filter and objects_req:
            candidates = self.object_filter.filter_candidates(candidates, objects_req, mode="boost")
        elif self.object_filter:
            labels = self.object_filter.get_all_labels()
            found = [label for label in labels if re.search(r"\b" + re.escape(label.lower()) + r"(?:s|es)?\b", english_question.lower())]
            if found:
                candidates = self.object_filter.filter_candidates(candidates, found, mode="boost")

        candidates = candidates[:top_k]
        if self.vlm_pipeline and config.VQA_VERIFY_CANDIDATES:
            for candidate in candidates:
                try:
                    candidate["verified"] = self.vlm_pipeline.verify_frame(
                        str(config.keyframe_path(candidate["video_id"], candidate["keyframe_name"])),
                        retrieval_desc,
                    )
                    if candidate["verified"]:
                        candidate["score"] = float(candidate.get("score", 0.0)) + config.VQA_VERIFICATION_BOOST
                except Exception as exc:
                    candidate["verified"] = True
                    candidate["verification_error"] = str(exc)
            candidates.sort(key=lambda item: item.get("score", 0.0), reverse=True)

        results = []
        for candidate in candidates[:top_k]:
            candidate = candidate.copy()
            candidate["answer_error"] = ""
            if self.vlm_pipeline:
                try:
                    candidate["answer"] = self.vlm_pipeline.answer_question(
                        str(config.keyframe_path(candidate["video_id"], candidate["keyframe_name"])),
                        vlm_question,
                    )
                except Exception as exc:
                    candidate["answer"] = ""
                    candidate["answer_error"] = str(exc)
            else:
                candidate["answer"] = ""
                candidate["answer_error"] = "VLM is not configured"
            results.append(candidate)
            if progress_callback:
                progress_callback(len(results), top_k)

        return results, analysis
