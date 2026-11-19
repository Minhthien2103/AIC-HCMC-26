"""Test actual API/formatting code without loading GPU models or third-party deps."""
import ast
import csv
import io
from pathlib import Path
from types import SimpleNamespace
import unittest

TREE = ast.parse((Path(__file__).resolve().parents[1] / "src/server.py").read_text(encoding="utf-8"))


class Table:
    def __init__(self, rows):
        self.rows = rows

    def filter(self, video):
        return Table([row for row in self.rows if row["video_id"] == video])

    def sort(self, field):
        return Table(sorted(self.rows, key=lambda row: row[field]))

    def is_empty(self):
        return not self.rows

    def iter_rows(self, named=True):
        return iter(self.rows)


class Column:
    def __eq__(self, value):
        return value


class HttpError(Exception):
    def __init__(self, status_code, detail):
        self.status_code = status_code


def compile_function(name, namespace):
    node = next(n for n in ast.walk(TREE) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
    node.decorator_list = []
    exec(compile("from __future__ import annotations\n" + ast.unparse(node), name, "exec"), namespace)
    return namespace[name]


def run_handler(coroutine):
    # This endpoint reads in-memory metadata and has no await points.
    # Avoid creating a Windows asyncio event loop/socket for a CPU-only test.
    try:
        coroutine.send(None)
    except StopIteration as result:
        return result.value
    finally:
        coroutine.close()
    raise AssertionError("Unexpected asynchronous dependency")


class FrameMetadataTests(unittest.TestCase):
    def test_search_preserves_distinct_ids_and_precise_timestamp(self):
        formatter = compile_function("format_results", {"normalized_score": lambda row: 0.5})
        result = formatter(None, [{"video_id": "L01_V001", "keyframe_name": "002",
                                  "frame_id": 309, "pts_time": 12.3456, "faiss_idx": 4}])
        entity = result[0]["entity"]
        self.assertEqual(entity["frame_id"], "002")
        self.assertEqual(entity["keyframe_name"], "002")
        self.assertEqual(entity["original_frame_id"], 309)
        self.assertEqual(entity["time"], 12.3456)

    def test_timeline_uses_same_metadata_and_handles_missing_video(self):
        rows = [{"video_id": "L01_V001", "keyframe_name": "002", "frame_id": 309, "pts_time": 12.3456},
                {"video_id": "L01_V001", "keyframe_name": "001", "frame_id": 0, "pts_time": 0},
                {"video_id": "L02_V001", "keyframe_name": "001", "frame_id": 900, "pts_time": 30}]
        service = SimpleNamespace(retriever=SimpleNamespace(meta_df=Table(rows)))
        handler = compile_function("frame_mapping", {"service": service, "pl": SimpleNamespace(col=lambda _: Column()),
                                                     "io": io, "csv": csv, "HTTPException": HttpError})
        data = list(csv.DictReader(io.StringIO(run_handler(handler("L01_V001")))))
        self.assertEqual(data, [{"n": "001", "pts_time": "0", "frame_idx": "0"},
                                {"n": "002", "pts_time": "12.3456", "frame_idx": "309"}])
        with self.assertRaises(HttpError) as caught:
            run_handler(handler("missing"))
        self.assertEqual(caught.exception.status_code, 404)
        service.retriever.meta_df = None
        with self.assertRaises(HttpError) as caught:
            run_handler(handler("L01_V001"))
        self.assertEqual(caught.exception.status_code, 503)


if __name__ == "__main__":
    unittest.main()
