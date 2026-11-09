from __future__ import annotations

import polars as pl

from src.online_pipeline.frame_neighborhood import FrameNeighborhood
from src.tasks.kis_t import KIStask


class _Encoder:
    text_context_length = 10

    def __init__(self):
        self.batches: list[list[str]] = []

    def encode_text(self, query: str):
        return query

    def encode_text_batch(self, queries: list[str]):
        self.batches.append(list(queries))
        return list(queries)


class _Retriever:
    index = type("_Index", (), {"ntotal": 12})()

    @staticmethod
    def _rows(offset: int = 0):
        return [
            {
                "faiss_idx": index,
                "video_id": f"V{index // 3}",
                "frame_id": index,
                "keyframe_name": f"{index:03d}",
                "score": 1 - (index + offset) / 100,
            }
            for index in range(12)
        ]

    def search(self, query_vector, top_k: int):
        if "special" in str(query_vector):
            return [self._rows()[7], *self._rows()[:top_k]][:top_k]
        return self._rows()[:top_k]

    def search_batch(self, query_vectors, top_k: int):
        return [self.search(vector, top_k) for vector in query_vectors]


class _SingleVideoRetriever(_Retriever):
    def _rows(self, offset: int = 0):
        return [
            {
                "faiss_idx": index,
                "video_id": "V0",
                "frame_id": index,
                "keyframe_name": f"{index:03d}",
                "score": 1 - (index + offset) / 100,
            }
            for index in range(12)
        ]


class _Qwen:
    def __init__(self, *, fail_analysis: bool = False):
        self.fail_analysis = fail_analysis
        self.calls: list[str] = []

    def analyze_kis_query(self, query: str):
        if self.fail_analysis:
            raise RuntimeError("model unavailable")
        return {
            "retrieval_queries": ["special visual target"],
            "must_have": ["a unique visible target"],
            "expansions": [],
        }

    def score_kis_match(self, image_path: str, original_query: str, must_have: list[str]):
        self.calls.append(image_path)
        return 3 if image_path.endswith("007.jpg") else 1


class _MediaRetriever:
    def search(self, query: str, top_k: int):
        del query, top_k
        return [{"video_id": "V0", "metadata_score": 0.99, "metadata_text": "matching video"}]


class _RejectingQwen:
    def score_kis_match(self, image_path: str, original_query: str, must_have: list[str]):
        del image_path, original_query, must_have
        return 0


def test_long_query_variants_keep_tail_and_fit_context():
    task = KIStask(_Encoder(), _Retriever())
    query = "one two three four five six seven. eight nine FINAL_TARGET_TOKEN ten eleven twelve."

    variants = task._query_variants(query)

    assert variants
    assert "FINAL_TARGET_TOKEN" in " ".join(variants)
    assert all(task._token_count(variant) <= task._context_limit() for variant in variants)


def test_rrf_rewards_support_from_multiple_variants():
    fused = KIStask._rrf_fuse([
        [{"faiss_idx": 1, "score": 0.99}, {"faiss_idx": 2, "score": 0.50}],
        [{"faiss_idx": 2, "score": 0.40}],
    ])

    assert fused[0]["faiss_idx"] == 2
    assert fused[0]["rrf_score"] > fused[1]["rrf_score"]


def test_kis_does_not_trigger_temporal_and_keeps_baseline(monkeypatch):
    task = KIStask(_Encoder(), _SingleVideoRetriever())
    monkeypatch.setattr("src.tasks.kis_t.translate_vi_to_en", lambda value: value)

    results = task.execute("A person walks, then opens a door.", top_k=8)

    assert len(results) == 8
    assert results[0]["faiss_idx"] == 0
    # The old hard cap of three frames per video is gone: KIS ranking retains
    # the best exact frame candidates from a single video.
    assert sum(result["video_id"] == "V0" for result in results) == 8


def test_qwen_is_an_equal_evidence_source_not_a_hard_promotion(monkeypatch):
    qwen = _Qwen()
    task = KIStask(_Encoder(), _Retriever(), vlm_pipeline=qwen, enable_qwen=True, vlm_top_k=4)
    monkeypatch.setattr("src.tasks.kis_t.translate_vi_to_en", lambda value: value)

    results = task.execute("ordinary query", top_k=5)

    # Qwen's own ranking is only an equal RRF source; it cannot promote a
    # candidate solely because a threshold was crossed.
    assert results[0]["faiss_idx"] == 0
    # The Qwen pool is stratified by video; it does not receive an implicit
    # right to inspect every candidate or hard-promote a score threshold.
    assert results[0]["faiss_idx"] != 7
    assert len(qwen.calls) == 4


def test_qwen_analysis_failure_falls_back_to_clip(monkeypatch):
    task = KIStask(
        _Encoder(),
        _Retriever(),
        vlm_pipeline=_Qwen(fail_analysis=True),
        enable_qwen=True,
        vlm_top_k=4,
    )
    monkeypatch.setattr("src.tasks.kis_t.translate_vi_to_en", lambda value: value)

    results = task.execute("ordinary query", top_k=5)

    # With no Qwen analysis, frame order follows video-first D'Hondt coverage
    # rather than the old all-frame global ranking.
    assert [candidate["faiss_idx"] for candidate in results] == [0, 3, 6, 9, 1]


def test_qwen_rejections_are_not_rank_fusion_evidence():
    ranked = KIStask._qwen_ranked(_Retriever._rows()[:3], "query", [], _RejectingQwen())

    assert ranked == []


def test_media_video_candidates_never_enter_frame_neighborhood(monkeypatch):
    metadata = pl.DataFrame({
        "video_id": ["V0"] * 12,
        "faiss_idx": list(range(12)),
        "frame_id": list(range(12)),
    })
    task = KIStask(
        _Encoder(),
        _SingleVideoRetriever(),
        media_retriever=_MediaRetriever(),
        frame_neighborhood=FrameNeighborhood(metadata),
    )
    monkeypatch.setattr("src.tasks.kis_t.translate_vi_to_en", lambda value: value)

    results = task.execute("ordinary query", top_k=5)

    assert len(results) == 5
    assert all(candidate.get("frame_id") is not None for candidate in results)
    assert all(candidate.get("keyframe_name") for candidate in results)


def test_simple_baseline_concentrates_top_five_on_strongest_videos(monkeypatch):
    task = KIStask(
        _Encoder(),
        _Retriever(),
        baseline_simple=True,
        video_budget=4,
        local_frame_budget=3,
    )
    monkeypatch.setattr("src.tasks.kis_t.translate_vi_to_en", lambda value: value)

    results = task.execute("ordinary query", top_k=8)

    assert len(results) == 8
    assert results[0]["video_id"] == "V0"
    assert sum(item["video_id"] == "V0" for item in results[:5]) == 3
    assert len({item["video_id"] for item in results[:5]}) == 3
