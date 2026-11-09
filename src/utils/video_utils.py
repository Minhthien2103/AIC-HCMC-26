import cv2
import os

def get_video_info(video_path: str):
    """Returns basic video information: fps, total_frames, width, height."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cap.release()
    return {
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height
    }

def extract_frames(video_path: str, frame_indices: list):
    """
    Generator that yields (frame_idx, frame) for specified frame indices.
    Sorted to ensure efficient sequential reading.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
        
    sorted_indices = sorted(frame_indices)
    
    for idx in sorted_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            yield idx, frame
            
    cap.release()
