import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO, filename="retrieval_faiss.log")

# Load models and data
try:
    text_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    text_embeddings = np.load("text_embeddings.npy")
    image_embeddings = np.load("image_embeddings.npy")
    with open("text_data.json", "r") as f:
        text_data = json.load(f)
    with open("image_data.json", "r") as f:
        image_data = json.load(f)
    logging.info(f"Loaded {len(text_data)} text and {len(image_data)} image entries")
except Exception as e:
    logging.error(f"Data loading failed: {e}")
    raise e

# Normalize embeddings
try:
    text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
    image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)
except Exception as e:
    logging.error(f"Embedding normalization failed: {e}")
    raise e

# Create FAISS indexes
try:
    text_index = faiss.IndexFlatIP(384)  # Inner product (cosine similarity)
    image_index = faiss.IndexFlatIP(512)
    text_index.add(text_embeddings)
    image_index.add(image_embeddings)
    logging.info("FAISS indexes created")
except Exception as e:
    logging.error(f"FAISS index creation failed: {e}")
    raise e

def search_faiss(query, k=5, modality="text"):
    try:
        query_emb = text_model.encode([query])[0]
        query_emb = query_emb / np.linalg.norm(query_emb)
        query_emb = np.expand_dims(query_emb, axis=0).astype(np.float32)
        index = text_index if modality == "text" else image_index
        data = text_data if modality == "text" else image_data
        distances, indices = index.search(query_emb, k)
        results = [{"data": data[i], "score": float(distances[0][j])} for j, i in enumerate(indices[0])]
        logging.info(f"Query: {query}, Modality: {modality}, Results: {results}")
        return results
    except Exception as e:
        logging.error(f"FAISS search failed for query '{query}': {e}")
        return []

# Load gold test set
try:
    with open("gold_test_set.json", "r") as f:
        test_set = json.load(f)
    logging.info(f"Loaded {len(test_set)} gold test questions")
except Exception as e:
    logging.error(f"Gold test set loading failed: {e}")
    raise e

# Process gold test set
results = []
benchmark = {
    "top_1_correct": 0,
    "top_3_correct": 0,
    "top_5_correct": 0,
    "rejection_correct": 0,
    "total_answerable": sum(1 for item in test_set if item["answerable"]),
    "total_unanswerable": sum(1 for item in test_set if not item["answerable"])
}

for item in test_set:
    query = item["query"]
    is_answerable = item["answerable"]
    ground_truth = item["ground_truth"]
    
    # Search with k=5 for benchmarking
    search_results = search_faiss(query, k=5, modality="text")
    
    # Top-1 result for display
    if search_results and search_results[0]["score"] > 0.7:
        top_1 = search_results[0]
        results.append({"query": query, "result": top_1, "answerable": True})
    else:
        results.append({"query": query, "result": "Unanswerable", "answerable": False})
    
    # Benchmarking
    if is_answerable:
        gt_start_time = ground_truth["start_time"]
        for k in [1, 3, 5]:
            top_k = search_results[:k]
            if any(abs(r["data"]["start_time"] - gt_start_time) < 10.0 for r in top_k):
                benchmark[f"top_{k}_correct"] += 1
    else:
        if not search_results or search_results[0]["score"] < 0.7:
            benchmark["rejection_correct"] += 1

# Calculate percentages
for k in [1, 3, 5]:
    benchmark[f"top_{k}_percent"] = (benchmark[f"top_{k}_correct"] / benchmark["total_answerable"]) * 100 if benchmark["total_answerable"] > 0 else 0
benchmark["rejection_percent"] = (benchmark["rejection_correct"] / benchmark["total_unanswerable"]) * 100 if benchmark["total_unanswerable"] > 0 else 0

# Save results and benchmark
try:
    with open("faiss_results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open("faiss_benchmark.json", "w") as f:
        json.dump(benchmark, f, indent=2)
    logging.info("FAISS results and benchmark saved")
    print("Results:", json.dumps(results, indent=2))
    print("Benchmark:", json.dumps(benchmark, indent=2))
except Exception as e:
    logging.error(f"Saving results failed: {e}")
    raise e