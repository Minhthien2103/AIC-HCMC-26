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
        video_ids = metadata_df["video_id"].unique().to_list()
        video_ids.sort()

        print(f"Building FAISS Index for {len(video_ids)} metadata")
        
        self.index = faiss.IndexFlatIP(self.config.CLIP_DIM)
        total_vectors = 0
        
        for vid in video_ids:
            npy_path = self.config.CLIP_FEATURES_DIR / f"{vid}.npy"
            if not npy_path.exists():
                print(f"Warning: {npy_path.name} not found. Skipped.")
                continue
                
            features = np.load(npy_path)
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            features = features.astype('float32')
            
            faiss.normalize_L2(features)
            self.index.add(features)
            
            total_vectors += features.shape[0]
            
        print(f"FAISS Index built successfully with {total_vectors} vectors.")
        return True
        
    def save(self):
        if self.index is not None:
            faiss.write_index(self.index, str(self.config.FAISS_INDEX_PATH))
            print(f"Saved FAISS index to {self.config.FAISS_INDEX_PATH}")

if __name__ == "__main__":
    builder = FAISSIndexBuilder(config)
    if builder.build_index():
        builder.save()