import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO, filename="retrieval_faiss.log")

# Load models and data
text_model = SentenceTransformer("all-MiniLM-L6-v2")
text_embeddings = np.load("text_embeddings.npy")
image_embeddings = np.load("image_embeddings.npy")
with open("text_data.json", "r") as f:
    text_data = json.load(f)
with open("image_data.json", "r") as f:
    image_data = json.load(f)

# Normalize embeddings
text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)

# Create FAISS indexes
text_index = faiss.IndexFlatIP(384)  # Inner product (cosine similarity)
image_index = faiss.IndexFlatIP(512)
text_index.add(text_embeddings)
image_index.add(image_embeddings)
logging.info("FAISS indexes created")

def search_faiss(query, k=3, modality="text"):
    try:
        query_emb = text_model.encode([query])[0]
        query_emb = query_emb / np.linalg.norm(query_emb)
        query_emb = np.expand_dims(query_emb, axis=0)
        if modality == "text":
            distances, indices = text_index.search(query_emb, k)
            results = [(text_data[i], float(distances[0][j])) for j, i in enumerate(indices[0])]
        else:  # image or combined
            distances, indices = image_index.search(query_emb, k)
            results = [(image_data[i], float(distances[0][j])) for j, i in enumerate(indices[0])]
        logging.info(f"Query: {query}, Results: {results}")
        return results
    except Exception as e:
        logging.error(f"FAISS search failed: {e}")
        return []

# Test
query = "What is combinatorial reconfiguration?"
results = search_faiss(query, k=3)
print(json.dumps(results, indent=2))