import sys
from pathlib import Path
import numpy as np
import faiss
import polars as pl

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config


class FAISSIndexBuilder:
    def __init__(self, config_module):
        self.config = config_module
        self.index = None

        
    def build_index(self):
        if not self.config.METADATA_PATH.exists():
            print("Not found metadata")
            return False

            
        metadata_df = pl.read_parquet(self.config.METADATA_PATH)
        video_ids = sorted(metadata_df["video_id"].unique().to_list())

        print(f"Building FAISS Index for {len(video_ids)} metadata")
        
        self.index = faiss.IndexFlatIP(self.config.CLIP_DIM)
        total_vectors = 0
        
        expected_total = metadata_df.height
        for vid in video_ids:
            npy_path = self.config.CLIP_FEATURES_DIR / f"{vid}.npy"
            if not npy_path.exists():
                raise FileNotFoundError(f"Missing CLIP feature file: {npy_path}")
                
            features = np.load(npy_path)
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            features = features.astype('float32')
            expected_rows = metadata_df.filter(pl.col("video_id") == vid).height
            if features.shape != (expected_rows, self.config.CLIP_DIM):
                raise ValueError(
                    f"{npy_path.name}: expected {(expected_rows, self.config.CLIP_DIM)}, got {features.shape}"
                )
            
            faiss.normalize_L2(features)
            self.index.add(features)
            
            total_vectors += features.shape[0]
            
        print(f"FAISS Index built successfully with {total_vectors} vectors.")
        if total_vectors != expected_total:
            raise ValueError(f"Built {total_vectors} vectors for {expected_total} metadata rows")
        return True
        
    def save(self):
        if self.index is not None:
            faiss.write_index(self.index, str(self.config.FAISS_INDEX_PATH))
            print(f"Saved FAISS index to {self.config.FAISS_INDEX_PATH}")

if __name__ == "__main__":
    builder = FAISSIndexBuilder(config)
    if builder.build_index():
        builder.save()
