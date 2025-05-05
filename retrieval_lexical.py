import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import logging

logging.basicConfig(level=logging.DEBUG, filename="retrieval_lexical.log")

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

def search_lexical(query, k=3, method="tfidf"):
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
        logging.error(f"Lexical search failed: {e}")
        return []

query = "What is combinatorial reconfiguration?"
results_tfidf = search_lexical(query, k=3, method="tfidf")
results_bm25 = search_lexical(query, k=3, method="bm25")
print("TF-IDF:", json.dumps(results_tfidf, indent=2))
print("BM25:", json.dumps(results_bm25, indent=2))