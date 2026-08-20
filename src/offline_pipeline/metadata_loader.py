import sys
import polars as pl
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config


class MetadataLoader:
    def __init__(self, config_module):
        self.config = config_module
        self.master_df = None
    
    def parse_csvs(self):
        csv_files = sorted(self.config.MAP_KEYFRAMES_DIR.glob("*.csv"))
        print(f"Building metadata from {len(csv_files)} .csv files")
        
        base_kf = str(self.config.KEYFRAMES_DIR).replace('\\', '/')
        base_vids = str(self.config.VIDEOS_DIR).replace('\\', '/')
        
        list_df = []

        for csv_path in csv_files:
            video_id = csv_path.stem
            group_name = video_id.split("_")[0]
            
            try:
                df = pl.read_csv(csv_path)
                df = df.with_columns([
                    pl.lit(video_id).alias("video_id"),
                    pl.col("n").cast(pl.Utf8).str.zfill(3).alias("keyframe_name"),
                    (pl.col("n") - 1).alias("npy_row_idx"),
                    pl.col("frame_idx").alias("frame_id"),
                ])
                df = df.with_columns([
                    pl.format(f"{base_kf}/{{}}/" + "{}.jpg", pl.col("video_id"), pl.col("keyframe_name")).alias("keyframe_path"),
                    pl.format(f"{base_vids}/video_{group_name}/{{}}.mp4", pl.col("video_id")).alias("video_path")
                ])
                df = df.select([
                    "video_id", "keyframe_name", "pts_time", "frame_id", "keyframe_path", "npy_row_idx", "video_path"
                ])

                list_df.append(df)

            except Exception as e:
                print(f"Error {csv_path.name}: {e}")
                
        if len(list_df) == 0:
            print("No metadata parsed.")
            return False
            
        self.master_df = pl.concat(list_df)
        if hasattr(self.master_df, "with_row_index"):
            self.master_df = self.master_df.with_row_index("faiss_idx")
        else:
            self.master_df = self.master_df.with_row_count("faiss_idx")
            
        return True
        
    def save(self):
        if self.master_df is not None:
            self.master_df.write_parquet(self.config.METADATA_PATH)
            print(f"Saved {self.master_df.height} keyframes to {self.config.METADATA_PATH}")

if __name__ == "__main__":
    loader = MetadataLoader(config)
    if loader.parse_csvs():
        loader.save()