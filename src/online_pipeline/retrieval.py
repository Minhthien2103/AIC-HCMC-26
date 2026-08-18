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
        print("Loading FAISS index and Metadata")

        try:
            if self.index_path.exists() and self.metadata_path.exists():
                self.index = faiss.read_index(str(self.index_path))
                self.meta_df = pl.read_parquet(self.metadata_path)
                if self.index.ntotal != self.meta_df.height:
                    raise ValueError(
                        f"FAISS ntotal ({self.index.ntotal}) != metadata rows ({self.meta_df.height})"
                    )
                if "faiss_idx" not in self.meta_df.columns:
                    raise ValueError("metadata.parquet is missing faiss_idx")
                expected = list(range(self.meta_df.height))
                actual = self.meta_df["faiss_idx"].cast(pl.Int64).to_list()
                if actual != expected:
                    raise ValueError("metadata faiss_idx must be contiguous 0..N-1")
            else:
                print(f"    Path not found: FAISS index ({self.index_path}) and Metadata ({self.metadata_path})")

        except Exception as e:
            print(f"    Loading FAISS, Metadata Error: {e}")
            self.index = None
            self.meta_df = None


    def search(self, query_vector: np.ndarray, top_k: int = 100) -> list[dict]:
        if self.index is None or self.meta_df is None:
            return []

        query_vector = query_vector.astype('float32').reshape(1, -1)
        if query_vector.shape[1] != self.index.d:
            raise ValueError(f"Query dimension {query_vector.shape[1]} != index dimension {self.index.d}")
        top_k = max(0, min(int(top_k), int(self.index.ntotal)))
        if top_k == 0:
            return []
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
        if self.index is None or self.meta_df is None:
            return[]

        vid_meta = self.meta_df.filter(pl.col("video_id") == video_id)
        if vid_meta.height == 0:
            print(f"Video {video_id} doesn't has metadata")
            return []

        faiss_indices = vid_meta["faiss_idx"].to_list()
        vid_vectors = np.array([self.index.reconstruct(i) for i in faiss_indices])

        query_vector = query_vector.astype("float32").reshape(-1)
        if query_vector.shape[0] != self.index.d:
            raise ValueError(f"Query dimension {query_vector.shape[0]} != index dimension {self.index.d}")
        scores = np.dot(vid_vectors, query_vector)
        top_k = max(0, min(int(top_k), vid_meta.height))
        top_indices = np.argsort(scores)[::-1][:top_k]

        list_result = []

        for i in top_indices:
            row = vid_meta.row(i, named = True)
            row["score"] = float(scores[i])
            list_result.append(row)

        return list_result
