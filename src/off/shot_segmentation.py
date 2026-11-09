import cv2
import numpy as np
from src.config import OPTICAL_FLOW_THRESHOLD
from src.utils.video_utils import get_video_info

class ShotSegmenter:
    """
    Handles dividing a video into coherent shots.
    For sousveillance (egocentric) data, we apply a motion filter 
    to prevent over-segmentation caused by camera shake.
    """
    def __init__(self, use_autoshot=True):
        self.use_autoshot = use_autoshot
        # TODO: Initialize actual AutoShot or TransNetV2 model here
        
    def detect_shots(self, video_path: str):
        """
        Main pipeline for shot boundary detection.
        1. Run base detector (AutoShot/TransNetV2)
        2. Apply optical flow motion filter for sousveillance data
        """
        print(f"Detecting shots for {video_path}...")
        
        # 1. Base prediction
        raw_boundaries = self._run_base_detector(video_path)
        
        # 2. Filter false cuts caused by camera shake
        filtered_boundaries = self._apply_motion_filter(video_path, raw_boundaries)
        
        print(f"Found {len(filtered_boundaries)} valid shots.")
        return filtered_boundaries

    def _run_base_detector(self, video_path: str):
        """
        Mock implementation of AutoShot / TransNetV2 output.
        In reality, this will run inference on the video.
        """
        info = get_video_info(video_path)
        total_frames = info["total_frames"]
        
        # Mocking 2 cuts for testing (e.g. at frame 100)
        # We will replace this with real model inference
        if total_frames > 100:
            return [(0, 100), (101, total_frames - 1)]
        else:
            return [(0, total_frames - 1)]

    def _apply_motion_filter(self, video_path: str, boundaries: list):
        """
        Checks adjacent shots. If the 'cut' is just severe camera shake 
        (high global flow, low semantic change), merge them.
        """
        if len(boundaries) <= 1:
            return boundaries
            
        merged = [boundaries[0]]
        cap = cv2.VideoCapture(video_path)
        
        for i in range(1, len(boundaries)):
            prev_shot = merged[-1]
            curr_shot = boundaries[i]
            
            # Get the frames right at the boundary
            end_of_prev = prev_shot[1]
            start_of_curr = curr_shot[0]
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, end_of_prev)
            ret1, frame1 = cap.read()
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_of_curr)
            ret2, frame2 = cap.read()
            
            should_merge = False
            if ret1 and ret2:
                # Convert to grayscale
                gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
                
                # Calculate dense optical flow (Farneback)
                # Parameters: pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags
                flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                mean_mag = np.mean(mag)
                
                # If the flow magnitude is very high, it might be a shaky camera fake-cut
                if mean_mag > OPTICAL_FLOW_THRESHOLD:
                    # TODO: Add a check for semantic similarity (e.g., embedding cosine distance)
                    # If semantics didn't change much despite the movement, merge the shots.
                    should_merge = True
            
            if should_merge:
                # Merge the shots
                merged[-1] = (prev_shot[0], curr_shot[1])
            else:
                merged.append(curr_shot)
                
        cap.release()
        return merged
