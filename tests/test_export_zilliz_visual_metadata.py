from pathlib import Path

import polars as pl
import pytest

from scripts.export_zilliz_visual_metadata import export_visual_metadata


class FakeIterator:
    def __init__(self, batches):
        self.batches = iter(batches)
        self.closed = False

    def next(self):
        return next(self.batches, [])

    def close(self):
        self.closed = True


class FakeClient:
    def __init__(self, rows):
        self.rows = rows
        self.iterator = None
        self.query_kwargs = None

    def list_collections(self):
        return ["visual"]

    def describe_collection(self, name):
        assert name == "visual"
        return {
            "fields": [
                {"name": name}
                for name in ("pk", "embedding", "video_id", "frame_name", "frame_idx", "timestamp")
            ]
        }

    def load_collection(self, name):
        assert name == "visual"

    def query_iterator(self, **kwargs):
        self.query_kwargs = kwargs
        self.iterator = FakeIterator([self.rows[:1], self.rows[1:], []])
        return self.iterator


def test_export_visual_metadata_uses_scalars_only(tmp_path: Path):
    client = FakeClient(
        [
            {
                "pk": 2,
                "video_id": "L21_V001",
                "frame_name": "frame_000020.jpg",
                "frame_idx": 20,
                "timestamp": 0.8,
            },
            {
                "pk": 1,
                "video_id": "L21_V001",
                "frame_name": "frame_000008.jpg",
                "frame_idx": 8,
                "timestamp": 0.32,
            },
        ]
    )
    output = tmp_path / "metadata.parquet"

    metadata = export_visual_metadata(
        client,
        collection_name="visual",
        pk_field="pk",
        output_path=output,
        batch_size=100,
    )

    assert output.is_file()
    assert client.iterator.closed
    assert "embedding" not in client.query_kwargs["output_fields"]
    assert metadata["frame_key"].to_list() == ["L21_V001::8", "L21_V001::20"]
    assert metadata["frame_order"].to_list() == [0, 1]
    assert pl.read_parquet(output).height == 2


def test_export_visual_metadata_rejects_duplicate_frame_keys(tmp_path: Path):
    row = {
        "pk": 1,
        "video_id": "L21_V001",
        "frame_name": "frame_000008.jpg",
        "frame_idx": 8,
        "timestamp": 0.32,
    }
    client = FakeClient([row, {**row, "pk": 2}])

    with pytest.raises(RuntimeError, match="frame key is not unique"):
        export_visual_metadata(
            client,
            collection_name="visual",
            pk_field="pk",
            output_path=tmp_path / "metadata.parquet",
        )
