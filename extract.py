import cv2
import os
import json
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, filename="keyframes.log")

# Paths (WSL-friendly)
video_path = os.path.join("input", "Video.mp4")
keyframes_dir = "/home/raph/keyframes"  # Avoid /mnt/c for reliability
os.makedirs(keyframes_dir, exist_ok=True)

# Check if video file exists
if not os.path.exists(video_path):
    logging.error(f"Video file not found at {video_path}")
    raise FileNotFoundError(f"Video file not found at {video_path}")

# Open the video
try:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Failed to open video: {video_path}")
        raise ValueError(f"Failed to open video: {video_path}")
except Exception as e:
    logging.error(f"Video capture error: {e}")
    raise e

# Scene detection parameters
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    logging.error("Invalid FPS value")
    raise ValueError("Invalid FPS value")
frame_count = 0
keyframes = []
prev_hist = None
threshold = 0.6  # Histogram difference threshold for slide change
min_interval = 30 * fps  # Minimum 30s between keyframes to avoid false positives
last_keyframe = -min_interval

# Fallback sampling (every 60 seconds)
fallback_interval = int(fps * 60)  # 60 seconds
next_fallback = fallback_interval

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    timestamp = frame_count / fps

    # Compute histogram for scene detection
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([frame_gray], [0], None, [256], [0, 256])
    hist = hist / hist.sum()  # Normalize

    # Detect slide change
    if prev_hist is not None and frame_count - last_keyframe >= min_interval:
        hist_diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
        if hist_diff < threshold:
            frame_filename = f"frame_{timestamp:.1f}.jpg"
            frame_path = os.path.join(keyframes_dir, frame_filename)
            try:
                cv2.imwrite(frame_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                keyframes.append({"timestamp": timestamp, "frame_path": frame_path})
                logging.info(f"Saved keyframe at slide change: {frame_path}")
                last_keyframe = frame_count
            except Exception as e:
                logging.error(f"Failed to save keyframe {frame_path}: {e}")

    # Fallback sampling
    if frame_count >= next_fallback:
        frame_filename = f"frame_{timestamp:.1f}.jpg"
        frame_path = os.path.join(keyframes_dir, frame_filename)
        try:
            cv2.imwrite(frame_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            keyframes.append({"timestamp": timestamp, "frame_path": frame_path})
            logging.info(f"Saved fallback keyframe: {frame_path}")
            last_keyframe = frame_count
            next_fallback += fallback_interval
        except Exception as e:
            logging.error(f"Failed to save keyframe {frame_path}: {e}")

    prev_hist = hist
    frame_count += 1

cap.release()

# Save keyframe metadata
keyframe_json_path = os.path.join(keyframes_dir, "keyframes.json")
try:
    with open(keyframe_json_path, "w") as f:
        json.dump(keyframes, f, indent=2)
    logging.info(f"Extracted {len(keyframes)} keyframes. Metadata saved to {keyframe_json_path}")
    print(f"Extracted {len(keyframes)} keyframes. Metadata saved to {keyframe_json_path}")
except Exception as e:
    logging.error(f"Failed to save keyframes.json: {e}")
    raise e

# Align keyframes with transcripts
try:
    with open("transcripts.json", "r") as f:
        transcripts = json.load(f)
    for kf in keyframes:
        transcript = next((t for t in transcripts if t["start_time"] <= kf["timestamp"] < t["end_time"]), None)
        if transcript:
            kf["transcript_text"] = transcript["text"]
    with open(os.path.join(keyframes_dir, "keyframes_mapped.json"), "w") as f:
        json.dump(keyframes, f, indent=2)
    logging.info("Aligned keyframes with transcripts")
except Exception as e:
    logging.error(f"Failed to align keyframes with transcripts: {e}")