import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import logging

logging.basicConfig(level=logging.DEBUG, filename="retrieval_lexical.log")

# Load data
try:
    with open("text_data.json", "r") as f:
        text_data = json.load(f)
    texts = [item["text"] for item in text_data if item["text"].strip()]
    logging.info(f"Loaded {len(texts)} text entries")
except Exception as e:
    logging.error(f"Data loading failed: {e}")
    raise e

# TF-IDF
try:
    tfidf_vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    logging.info("TF-IDF matrix created")
except Exception as e:
    logging.error(f"TF-IDF setup failed: {e}")
    raise e

# BM25
try:
    tokenized_texts = [text.lower().split() for text in texts]
    bm25 = BM25Okapi(tokenized_texts)
    logging.info("BM25 index created")
except Exception as e:
    logging.error(f"BM25 setup failed: {e}")
    raise e

def search_lexical(query, k=5, method="tfidf"):
    try:
        if method == "tfidf":
            query_vec = tfidf_vectorizer.transform([query])
            scores = (tfidf_matrix * query_vec.T).toarray().flatten()
        else:  # bm25
            tokenized_query = query.lower().split()
            scores = bm25.get_scores(tokenized_query)
        top_k = np.argsort(scores)[::-1][:k]
        results = [{"data": text_data[i], "score": float(scores[i])} for i in top_k]
        logging.debug(f"Query: {query}, Method: {method}, Results: {results}")
        return results
    except Exception as e:
        logging.error(f"Lexical search failed for query '{query}': {e}")
        return []

# Load gold test set
try:
    with open("gold_test_set.json", "r") as f:
        test_set = json.load(f)
    logging.info(f"Loaded {len(test_set)} gold test questions")
except Exception as e:
    logging.error(f"Gold test set loading failed: {e}")
    raise e

# Process gold test set for TF-IDF and BM25
for method in ["tfidf", "bm25"]:
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
        search_results = search_lexical(query, k=5, method=method)
        
        # Top-1 result for display
        if search_results and search_results[0]["score"] > 0.5:
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
            if not search_results or search_results[0]["score"] < 0.5:
                benchmark["rejection_correct"] += 1

    # Calculate percentages
    for k in [1, 3, 5]:
        benchmark[f"top_{k}_percent"] = (benchmark[f"top_{k}_correct"] / benchmark["total_answerable"]) * 100 if benchmark["total_answerable"] > 0 else 0
    benchmark["rejection_percent"] = (benchmark["rejection_correct"] / benchmark["total_unanswerable"]) * 100 if benchmark["total_unanswerable"] > 0 else 0

    # Save results and benchmark
    try:
        with open(f"{method}_results.json", "w") as f:
            json.dump(results, f, indent=2)
        with open(f"{method}_benchmark.json", "w") as f:
            json.dump(benchmark, f, indent=2)
        logging.info(f"{method.upper()} results and benchmark saved")
        print(f"{method.upper()} Results:", json.dumps(results, indent=2))
        print(f"{method.upper()} Benchmark:", json.dumps(benchmark, indent=2))
    except Exception as e:
        logging.error(f"Saving {method} results failed: {e}")
        raise e