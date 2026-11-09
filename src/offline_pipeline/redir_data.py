import os
import shutil
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

def reorganize_data():
    base_dir = config.KEYFRAMES_DIR
    jpeg_dir = getattr(config, "JPEG_SIZE_DIR", config.DATA_DIR / "jpeg_sizes")
    jpeg_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Scanning {base_dir} for nested video folders...")
    
    npy_files = list(base_dir.rglob("jpeg_sizes.npy"))
    
    if not npy_files:
        print("No jpeg_sizes.npy found. Are they extracted into data/keyframes/ yet?")
        return

    count_npy = 0
    count_kf = 0
    skipped = 0
    
    for npy_path in npy_files:
        video_dir = npy_path.parent
        video_id = video_dir.name  # e.g., "L21_V001"
        
        target_npy = jpeg_dir / f"{video_id}.npy"
        target_kf_dir = base_dir / video_id
        nested_kf_dir = video_dir / "keyframes"
        
        # Skip if already organized
        if target_npy.exists() and target_kf_dir.exists():
            print(f"  [Skip] {video_id} already organized.")
            skipped += 1
            continue
            
        # 1. Move the npy file
        try:
            shutil.move(str(npy_path), str(target_npy))
            count_npy += 1
        except Exception as e:
            print(f"Error moving {npy_path.name}: {e}")
        
        # 2. Move all keyframes
        if nested_kf_dir.exists() and nested_kf_dir.is_dir():
            target_kf_dir.mkdir(parents=True, exist_ok=True)
            for img_file in nested_kf_dir.glob("*.jpg"):
                try:
                    shutil.move(str(img_file), str(target_kf_dir / img_file.name))
                    count_kf += 1
                except Exception as e:
                    pass
                    
    print(f"Successfully moved {count_npy} .npy files to {jpeg_dir}")
    print(f"Successfully moved {count_kf} keyframes to {base_dir}")
    print(f"Skipped {skipped} videos that were already organized.")
    
    # 3. Cleanup: Delete the old dake_L* folders since we extracted everything
    for dake_dir in base_dir.glob("dake_L*"):
        if dake_dir.is_dir():
            try:
                shutil.rmtree(str(dake_dir))
                print(f"Cleaned up empty directory tree: {dake_dir.name}")
            except Exception as e:
                print(f"Warning: could not delete {dake_dir.name} - {e}")

if __name__ == "__main__":
    reorganize_data()