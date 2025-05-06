import psycopg2
import numpy as np
import json
from sentence_transformers import SentenceTransformer
import logging
import torch

logging.basicConfig(level=logging.DEBUG, filename="retrieval_pgvector.log")

# Force CPU
torch.set_default_device("cpu")
logging.info("Set PyTorch to use CPU")

# Initialize database connection
def connect_to_db(dbname):
    try:
        conn = psycopg2.connect(
            dbname=dbname,
            user="postgres",
            password="postgres123",
            host="localhost",
            port="5432"
        )
        return conn
    except Exception as e:
        logging.error(f"Connection to {dbname} failed: {e}")
        raise e

# Recreate rag database
try:
    conn = connect_to_db("postgres")
    conn.set_session(autocommit=True)
    cur = conn.cursor()
    cur.execute("SELECT pg_terminate_backend(pg_stat_activity.pid) FROM pg_stat_activity WHERE pg_stat_activity.datname = 'rag' AND pid <> pg_backend_pid()")
    cur.execute("DROP DATABASE IF EXISTS rag")
    cur.execute("CREATE DATABASE rag")
    logging.info("Recreated rag database")
    cur.close()
    conn.close()
except Exception as e:
    logging.error(f"Database recreation failed: {e}")
    raise e

# Connect to rag database
try:
    conn = connect_to_db("rag")
    cur = conn.cursor()
    logging.info("Connected to rag database")
except Exception as e:
    logging.error(f"Connection to rag database failed: {e}")
    raise e

# Enable pgvector
try:
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()
    logging.info("pgvector extension enabled")
except Exception as e:
    logging.error(f"pgvector extension failed: {e}")
    raise e

# Create tables
try:
    cur.execute("""
        CREATE TABLE text_embeddings (
            id SERIAL PRIMARY KEY,
            start_time FLOAT,
            end_time FLOAT,
            text TEXT,
            embedding VECTOR(384)
        );
        CREATE TABLE image_embeddings (
            id SERIAL PRIMARY KEY,
            timestamp FLOAT,
            frame_path TEXT,
            transcript_text TEXT,
            embedding VECTOR(512)
        );
    """)
    conn.commit()
    logging.info("Database tables created")
except Exception as e:
    logging.error(f"Table creation failed: {e}")
    raise e

# Load and normalize data
try:
    text_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    with open("text_data.json", "r") as f:
        text_data = json.load(f)
    with open("image_data.json", "r") as f:
        image_data = json.load(f)
    valid_text_data = []
    for item in text_data:
        emb = np.array(item["embedding"])
        if len(emb) != 384:
            logging.warning(f"Skipping text embedding with length {len(emb)}")
            continue
        item["embedding"] = (emb / np.linalg.norm(emb)).tolist()
        valid_text_data.append(item)
    valid_image_data = []
    for item in image_data:
        emb = np.array(item["embedding"])
        if len(emb) != 512:
            logging.warning(f"Skipping image embedding with length {len(emb)}")
            continue
        item["embedding"] = (emb / np.linalg.norm(emb)).tolist()
        valid_image_data.append(item)
    logging.info(f"Loaded {len(valid_text_data)} text and {len(valid_image_data)} image entries")
except Exception as e:
    logging.error(f"Data loading failed: {e}")
    raise e

# Insert embeddings
try:
    cur.execute("DELETE FROM text_embeddings; DELETE FROM image_embeddings;")
    for item in valid_text_data:
        cur.execute(
            "INSERT INTO text_embeddings (start_time, end_time, text, embedding) VALUES (%s, %s, %s, %s)",
            (item["start_time"], item["end_time"], item["text"], item["embedding"])
        )
    for item in valid_image_data:
        cur.execute(
            "INSERT INTO image_embeddings (timestamp, frame_path, transcript_text, embedding) VALUES (%s, %s, %s, %s)",
            (item["timestamp"], item["frame_path"], item["transcript_text"], item["embedding"])
        )
    conn.commit()
    cur.execute("SELECT COUNT(*) FROM text_embeddings")
    text_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM image_embeddings")
    image_count = cur.fetchone()[0]
    logging.info(f"Inserted {text_count} text and {image_count} image embeddings")
except Exception as e:
    logging.error(f"Embedding insertion failed: {e}")
    raise e

# Create indexes
try:
    cur.execute("CREATE INDEX text_ivfflat ON text_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);")
    cur.execute("CREATE INDEX text_hnsw ON text_embeddings USING hnsw (embedding vector_cosine_ops) WITH (m = 24, ef_construction = 100);")
    cur.execute("CREATE INDEX image_ivfflat ON image_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);")
    cur.execute("CREATE INDEX image_hnsw ON image_embeddings USING hnsw (embedding vector_cosine_ops) WITH (m = 24, ef_construction = 100);")
    conn.commit()
    logging.info("pgvector indexes created")
except Exception as e:
    logging.error(f"Index creation failed: {e}")
    raise e

def search_pgvector(query, k=5, modality="text", index_type="hnsw"):
    try:
        query_emb = text_model.encode([query])[0]
        query_emb = query_emb / np.linalg.norm(query_emb)
        query_emb_str = str(query_emb.tolist())
        table = "text_embeddings" if modality == "text" else "image_embeddings"
        cur.execute(f"SET enable_seqscan = OFF; SET ivfflat.probes = 20; SET hnsw.ef_search = 200;")
        cur.execute(
            f"SELECT id, start_time, end_time, text, 1 - (embedding <=> %s::vector) AS score FROM {table} ORDER BY score DESC LIMIT %s",
            (query_emb_str, k)
        ) if modality == "text" else cur.execute(
            f"SELECT id, timestamp, frame_path, transcript_text, 1 - (embedding <=> %s::vector) AS score FROM {table} ORDER BY score DESC LIMIT %s",
            (query_emb_str, k)
        )
        results = [{"data": dict(zip([desc[0] for desc in cur.description], row)), "score": row[-1]} for row in cur.fetchall()]
        logging.debug(f"Query: {query}, Modality: {modality}, Index: {index_type}, Results: {results}")
        return results
    except Exception as e:
        logging.error(f"pgvector search failed for query '{query}': {e}")
        return []
# Load gold test set
try:
    with open("gold_test_set.json", "r") as f:
        test_set = json.load(f)
    logging.info(f"Loaded {len(test_set)} gold test questions")
except Exception as e:
    logging.error(f"Gold test set loading failed: {e}")
    raise e
# Process gold test set for IVFFlat and HNSW
for index_type in ["ivfflat", "hnsw"]:
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
        search_results = search_pgvector(query, k=5, index_type=index_type)
        
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
        with open(f"pgvector_{index_type}_results.json", "w") as f:
            json.dump(results, f, indent=2)
        with open(f"pgvector_{index_type}_benchmark.json", "w") as f:
            json.dump(benchmark, f, indent=2)
        logging.info(f"pgvector {index_type} results and benchmark saved")
        print(f"pgvector {index_type} Results:", json.dumps(results, indent=2))
        print(f"pgvector {index_type} Benchmark:", json.dumps(benchmark, indent=2))
    except Exception as e:
        logging.error(f"Saving pgvector {index_type} results failed: {e}")
        raise e

# Close connection
try:
    cur.close()
    conn.close()
    logging.info("Database connection closed")
except Exception as e:
    logging.error(f"Closing database connection failed: {e}")