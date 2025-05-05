import cv2
import json
import logging
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import requests
import time

# Setup logging
logging.basicConfig(level=logging.INFO, filename="/home/raph/embeddings.log")

transcripts_path = "transcripts.json"
keyframes_path = "keyframes/keyframes_mapped.json"
text_emb_path = "/home/raph/text_embeddings.npy"
image_emb_path = "/home/raph/image_embeddings.npy"
text_data_path = "/home/raph/text_data.json"
image_data_path = "/home/raph/image_data.json"

# Retry decorator for network requests
def retry(func, max_attempts=3, delay=5):
    def wrapper(*args, **kwargs):
        for attempt in range(max_attempts):
            try:
                return func(*args, **kwargs)
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                logging.warning(f"Attempt {attempt+1}/{max_attempts} failed: {e}")
                if attempt < max_attempts - 1:
                    time.sleep(delay)
                else:
                    raise e
    return wrapper

# Load models with retry
@retry
def load_clip_model():
    return CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

@retry
def load_clip_processor():
    return CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

try:
    text_model = SentenceTransformer("all-MiniLM-L6-v2")
    device = "cpu"  # Force CPU execution
    clip_model = load_clip_model().to(device)
    clip_processor = load_clip_processor()
    logging.info("Loaded text and image embedding models on CPU")
except Exception as e:
    logging.error(f"Failed to load models: {e}")
    raise e

# Load data
try:
    with open(transcripts_path, "r") as f:
        transcripts = json.load(f)
    with open(keyframes_path, "r") as f:
        keyframes = json.load(f)
    logging.info(f"Loaded {len(transcripts)} transcripts and {len(keyframes)} keyframes")
except Exception as e:
    logging.error(f"Failed to load data: {e}")
    raise e

# Text embeddings
try:
    texts = [t["text"] for t in transcripts if t["text"]]
    if not texts:
        raise ValueError("No valid transcript texts found")
    text_embeddings = text_model.encode(texts, show_progress_bar=True, batch_size=32)
    text_data = [
        {"start_time": t["start_time"], "end_time": t["end_time"], "text": t["text"], "embedding": emb.tolist()}
        for t, emb in zip(transcripts, text_embeddings)
        if t["text"]
    ]
    np.save(text_emb_path, text_embeddings)
    with open(text_data_path, "w") as f:
        json.dump(text_data, f, indent=2)
    logging.info(f"Saved {len(text_embeddings)} text embeddings")
except Exception as e:
    logging.error(f"Text embedding generation failed: {e}")
    raise e

# Image embeddings
try:
    image_embeddings = []
    for kf in keyframes:
        try:
            img = cv2.imread(kf["frame_path"])
            if img is None:
                logging.warning(f"Failed to load image: {kf['frame_path']}")
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            inputs = clip_processor(images=img_rgb, return_tensors="pt").to(device)
            with torch.no_grad():
                emb = clip_model.get_image_features(**inputs).cpu().numpy().flatten()
            image_embeddings.append(emb)
            logging.info(f"Generated embedding for {kf['frame_path']}")
        except Exception as e:
            logging.warning(f"Failed to process image {kf['frame_path']}: {e}")
            continue
    if not image_embeddings:
        raise ValueError("No valid image embeddings generated")
    image_embeddings = np.array(image_embeddings)
    image_data = [
        {"timestamp": kf["timestamp"], "frame_path": kf["frame_path"], "transcript_text": kf.get("transcript_text", ""), "embedding": emb.tolist()}
        for kf, emb in zip(keyframes, image_embeddings)
    ]
    np.save(image_emb_path, image_embeddings)
    with open(image_data_path, "w") as f:
        json.dump(image_data, f, indent=2)
    logging.info(f"Saved {len(image_embeddings)} image embeddings")
except Exception as e:
    logging.error(f"Image embedding generation failed: {e}")
    raise e

print(f"Generated {len(text_embeddings)} text embeddings and {len(image_embeddings)} image embeddings")