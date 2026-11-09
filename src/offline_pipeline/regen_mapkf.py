import os
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

def generate_maps():
    kf_dir = config.KEYFRAMES_DIR
    map_dir = getattr(config, "MAP_KEYFRAMES_DIR", config.DATA_DIR / "map-keyframes")
    map_dir.mkdir(parents=True, exist_ok=True)
    
    fps = 30.0 # Defaulting to 30.0 based on old CSVs
    videos = [d for d in kf_dir.iterdir() if d.is_dir()]
    
    count = 0
    for video_dir in videos:
        video_id = video_dir.name
        
        # Find all .jpg images
        images = list(video_dir.glob("*.jpg"))
        if not images:
            continue
                
        rows = []
        for i, img in enumerate(images):
            name = img.stem
            # Extract number from "frame_000080"
            try:
                frame_idx = int(''.join(filter(str.isdigit, name)))
            except ValueError:
                frame_idx = i * int(fps) 
                
            pts_time = round(frame_idx / fps, 4)
            rows.append({
                "n": 0, 
                "pts_time": pts_time,
                "fps": fps,
                "frame_idx": frame_idx
            })
            
        # Sort by frame_idx ascending
        rows.sort(key=lambda x: x["frame_idx"])
        
        # Assign 'n' index
        for i, r in enumerate(rows):
            r["n"] = i + 1
            
        df = pd.DataFrame(rows)
        csv_path = map_dir / f"{video_id}.csv"
        df.to_csv(csv_path, index=False, columns=["n", "pts_time", "fps", "frame_idx"])
        count += 1
        
    print(f"Successfully generated {count} map-keyframes CSVs in {map_dir}")

if __name__ == "__main__":
    generate_maps()