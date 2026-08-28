from __future__ import annotations

import numpy as np
import polars as pl

from src.online_pipeline.zilliz_object_filter import ZillizObjectFilter
from src.online_pipeline.zilliz_retrieval import ZillizRetrievalEngine


class FakeZillizClient:
    def __init__(self):
        self.search_calls = []
        self.query_calls = []

    def list_collections(self):
        return ["visual", "text", "objects"]

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        if kwargs["collection_name"] == "visual":
            hits = [
                {
                    "id": "v1",
                    "distance": 0.9,
                    "entity": {
                        "pk": "v1",
                        "video_id": "L25_V001",
                        "frame_name": "000100.jpg",
                        "frame_idx": 100,
                        "timestamp": 4.0,
                    },
                },
                {
                    "id": "v2",
                    "distance": 0.8,
                    "entity": {
                        "pk": "v2",
                        "video_id": "L25_V001",
                        "frame_name": "000200.jpg",
                        "frame_idx": 200,
                        "timestamp": 8.0,
                    },
                },
            ]
        else:
            hits = [
                {
                    "id": "t1",
                    "distance": 0.95,
                    "entity": {
                        "pk": "t1",
                        "video_id": "L25_V001",
                        "frame_number_start": 180,
                        "frame_number_end": 210,
                        "timestamp_start": 7.0,
                        "timestamp_end": 9.0,
                        "subtitles": "sealed boxes on a truck",
                    },
                }
            ]
        return [hits for _ in kwargs["data"]]

    def query(self, **kwargs):
        self.query_calls.append(kwargs)
        return [
            {"frame_key": "L25_V001::200", "class_name": "Person"},
            {"frame_key": "L25_V001::200", "class_name": "Box"},
        ]


def _metadata(path):
    pl.DataFrame(
        {
            "frame_key": ["L25_V001::100", "L25_V001::200"],
            "visual_pk": ["v1", "v2"],
            "video_id": ["L25_V001", "L25_V001"],
            "frame_name": ["000100.jpg", "000200.jpg"],
            "frame_idx": [100, 200],
            "timestamp": [4.0, 8.0],
            "object_class_names": [["Person"], ["Person", "Box"]],
            "text_pks": [[], ["t1"]],
        }
    ).write_parquet(path)


def test_zilliz_visual_and_subtitle_results_map_to_canonical_frames(tmp_path):
    metadata_path = tmp_path / "metadata.parquet"
    _metadata(metadata_path)
    client = FakeZillizClient()
    retriever = ZillizRetrievalEngine(
        uri="unused",
        token="unused",
        metadata_path=metadata_path,
        visual_collection="visual",
        text_collection="text",
        client=client,
    )

    rows = retriever.search(np.ones(512, dtype=np.float32), top_k=2)

    assert rows[0]["frame_key"] == "L25_V001::200"
    assert rows[0]["keyframe_name"] == "000200.jpg"
    assert rows[0]["frame_id"] == 200
    assert rows[0]["source_ranks"] == {"zilliz_subtitle": 1, "zilliz_visual": 2}
    assert retriever.index.ntotal == 2


def test_zilliz_video_filter_and_object_evidence(tmp_path):
    metadata_path = tmp_path / "metadata.parquet"
    _metadata(metadata_path)
    client = FakeZillizClient()
    retriever = ZillizRetrievalEngine(
        uri="unused",
        token="unused",
        metadata_path=metadata_path,
        visual_collection="visual",
        text_collection="text",
        client=client,
    )
    rows = retriever.search_in_video(
        np.ones(512, dtype=np.float32), video_id="L25_V001", top_k=2
    )
    assert all(call["filter"] == 'video_id == "L25_V001"' for call in client.search_calls)

    object_filter = ZillizObjectFilter(client, "objects", retriever.meta_df)
    matched = object_filter.matching_candidates(rows, ["person"], require_all=False)

    assert [row["frame_key"] for row in matched] == ["L25_V001::200"]
    assert matched[0]["matched_objects"] == ["person"]
    assert object_filter.get_all_labels() == ["Box", "Person"]
    assert client.query_calls[0]["collection_name"] == "objects"


def test_zilliz_skips_candidates_without_local_keyframes(tmp_path):
    metadata_path = tmp_path / "metadata.parquet"
    _metadata(metadata_path)
    video_dir = tmp_path / "keyframes" / "L25_V001"
    video_dir.mkdir(parents=True)
    (video_dir / "000200.jpg").touch()

    client = FakeZillizClient()
    retriever = ZillizRetrievalEngine(
        uri="unused",
        token="unused",
        metadata_path=metadata_path,
        visual_collection="visual",
        text_collection="text",
        keyframes_dir=tmp_path / "keyframes",
        client=client,
    )

    rows = retriever.search(np.ones(512, dtype=np.float32), top_k=2)

    assert [row["frame_key"] for row in rows] == ["L25_V001::200"]
    assert all(call["limit"] == 2 for call in client.search_calls)
