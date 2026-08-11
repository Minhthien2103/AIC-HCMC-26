import sys
import json
from pathlib import Path
import polars as pl
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

class ObjectStoreBuilder:
    def __init__(self, config_module):
        self.config = config_module
        self.objects_df = None
        
    def parse_jsons(self):
        if not self.config.METADATA_PATH.exists():
            print("Not found metadata")
            return False
        
        metadata_df = pl.read_parquet(self.config.METADATA_PATH)
        list_df = []
        cnt_failed = 0
        
        print(f"Building Object Store for {metadata_df.height} metadata")

        for row in tqdm(metadata_df.iter_rows(named=True), total = metadata_df.height, desc = "Parsing JSONs"):
            f_idx = row["faiss_idx"]
            vid = row["video_id"]
            k_name = row["keyframe_name"]
            
            json_path = self.config.OBJECTS_DIR / vid / f"{k_name}.json"
            if not json_path.exists():
                continue
                
            try:
                with open(json_path, 'r', encoding = 'utf-8') as f:
                    data = json.load(f)
                    
                scores = data.get("detection_scores", [])
                labels = data.get("detection_class_entities", [])
                
                for score_str, label in zip(scores, labels):
                    score = float(score_str)
                    if score >= self.config.OBJECT_CONF_THRESH:
                        list_df.append({
                            "faiss_idx": f_idx,
                            "object_label": label,
                            "score": score
                        })

                    else:
                        # print(f"Object {label} ({vid}, {k_name}.json), score: {score} < {self.config.OBJECT_CONF_THRESH}")
                        cnt_failed += 1

            except Exception:
                pass
                
        if len(list_df) == 0:
            print(f"No object has conf score > {self.config.OBJECT_CONF_THRESH}")
            return False
        else:
            print(f"{cnt_failed} object(s) has conf score > {self.config.OBJECT_CONF_THRESH}")

        self.objects_df = pl.DataFrame(list_df)
        return True
        
    def save(self):
        if self.objects_df is not None:
            self.objects_df.write_parquet(self.config.OBJECTS_PATH)
            print(f"Saved {self.objects_df.height} objects to {self.config.OBJECTS_PATH}")

if __name__ == "__main__":
    builder = ObjectStoreBuilder(config)
    if builder.parse_jsons():
        builder.save()