from __future__ import annotations

from src.tasks.vqa import VQATask


def _frame(video_id: str, frame_id: int, faiss_idx: int) -> dict:
    return {
        "video_id": video_id,
        "frame_id": frame_id,
        "keyframe_name": f"{frame_id:03d}",
        "faiss_idx": faiss_idx,
        "score": 1.0,
    }


class _Encoder:
    def encode_text_batch(self, values):
        return list(values)


class _Retriever:
    frames = [_frame("V1", 10, 0), _frame("V2", 20, 1)]

    def search_batch(self, vectors, top_k):
        return [self.frames[:top_k] for _value in vectors]

    def search_in_video_batch(self, vectors, video_id, top_k):
        rows = [item for item in self.frames if item["video_id"] == video_id][:top_k]
        return [rows for _value in vectors]


class _VLM:
    def __init__(self, details):
        self.details = details

    def analyze_vqa_query(self, question):
        return {"retrieval_description": question}

    def answer_question_details(self, image_path, question):
        return self.details


def test_vqa_uses_non_empty_low_confidence_answers_instead_of_aborting(monkeypatch):
    monkeypatch.setattr("src.tasks.vqa.translate_vi_to_en", lambda value: value)
    task = VQATask(
        _Encoder(),
        _Retriever(),
        vlm_pipeline=_VLM({"answer": "4", "visible_evidence": [], "confidence": 0}),
        qwen_candidate_budget=2,
    )

    results, _analysis = task.execute("count", top_k=2)

    assert len(results) == 2
    assert {item["answer"] for item in results} == {"4"}
    assert {item["answer_confidence"] for item in results} == {0}


def test_vqa_emits_valid_retrieval_fallback_when_every_answer_is_empty(monkeypatch):
    monkeypatch.setattr("src.tasks.vqa.translate_vi_to_en", lambda value: value)
    task = VQATask(
        _Encoder(),
        _Retriever(),
        vlm_pipeline=_VLM({"answer": "", "visible_evidence": [], "confidence": 0}),
        qwen_candidate_budget=2,
    )

    results, _analysis = task.execute("count", top_k=2)

    assert len(results) == 1
    assert results[0]["answer"] == "Không xác định"
    assert results[0]["answer_confidence"] == 0
