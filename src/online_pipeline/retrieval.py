import sys
import faiss
import numpy as np
import polars as pl
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))


class RetrievalEngine:
    def __init__(self, index_path: Path, metadata_path: Path):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = None
        self.meta_df = None

        self._load_engine()

    def _load_engine(self):
        print(f"Loading FAISS index and Metadata")

        try:
            if self.index_path.exists() and self.metadata_path.exists():
                self.index = faiss.read_index(str(self.index_path))
                self.meta_df = pl.read_parquet(self.metadata_path)
            else:
                print(f"    Path not found: FAISS index ({self.index_path}) and Metadata ({self.metadata_path})")

        except Exception as e:
            print(f"    Loading FAISS, Metadata Error: {e}")


    def search(self, query_vector: np.ndarray, top_k: int = 100) -> list[dict]:
        if self.index is None and self.meta_df is None:
            return []

        query_vector = query_vector.astype('float32').reshape(1, -1)
        scores, faiss_indices = self.index.search(query_vector, top_k)

        list_result = []

        for i in range(top_k):
            f_idx = int(faiss_indices[0][i])
            score = float(scores[0][i])

            if f_idx == -1:
                continue

            row = self.meta_df.row(f_idx, named = True)
            row["score"] = score

            list_result.append(row)

        return list_result


    def search_in_video(self, query_vector: np.ndarray, video_id: str, top_k: int = 5) -> list[dict]:
        if self.index is None and self.meta_df is None:
            return[]

        vid_meta = self.meta_df.filter(pl.col("video_id") == video_id)
        if vid_meta.height == 0:
            print(f"Video {video_id} doesn't has metadata")

        faiss_indices = vid_meta["faiss_idx"].to_list()
        vid_vectors = np.array([self.index.reconstruct(i) for i in faiss_indices])

        scores = np.dot(vid_vectors, query_vector)
        top_indices = np.argsort(scores)[::-1][:top_k]

        list_result = []

        for i in top_indices:
            row = vid_meta.row(i, named = True)
            row["score"] = float(scores[i])
            list_result.append(row)

        return list_result
    