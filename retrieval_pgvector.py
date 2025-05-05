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

# Connect to postgres database to manage rag database
try:
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="postgres123",
        host="localhost",
        port="5432"
    )
    conn.set_session(autocommit=True)
    cur = conn.cursor()
    logging.info("Connected to postgres database")
except Exception as e:
    logging.error(f"Connection to postgres database failed: {e}")
    raise e

# Drop and recreate rag database
try:
    cur.execute("SELECT pg_terminate_backend(pg_stat_activity.pid) FROM pg_stat_activity WHERE pg_stat_activity.datname = 'rag' AND pid <> pg_backend_pid()")
    cur.execute("DROP DATABASE IF EXISTS rag")
    cur.execute("CREATE DATABASE rag")
    logging.info("Recreated rag database")
except Exception as e:
    logging.error(f"Database recreation failed: {e}")
    raise e
finally:
    cur.close()
    conn.close()

# Connect to rag database
try:
    conn = psycopg2.connect(
        dbname="rag",
        user="postgres",
        password="postgres123",
        host="localhost",
        port="5432"
    )
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

def search_pgvector(query, k=3, modality="text", index_type="hnsw"):
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
        logging.debug(f"Query: {query}, Modality: {modality}, Index: {index_type}, Query embedding: {query_emb[:5]}..., Results: {results}")
        return results
    except Exception as e:
        logging.error(f"pgvector search failed: {e}")
        return []

# Test
try:
    cur.execute("SELECT COUNT(*) FROM text_embeddings")
    text_count = cur.fetchone()[0]
    if text_count == 0:
        raise ValueError("No text embeddings in database")
    cur.execute("SELECT COUNT(*) FROM image_embeddings")
    image_count = cur.fetchone()[0]
    logging.info(f"Text embeddings count: {text_count}, Image embeddings count: {image_count}")
    cur.execute("SELECT id, start_time, text FROM text_embeddings WHERE text ILIKE '%combinatorial reconfiguration%'")
    segment = cur.fetchall()
    logging.info(f"Found {len(segment)} segments with 'combinatorial reconfiguration': {segment}")
    query = "What is combinatorial reconfiguration?"
    results_ivfflat = search_pgvector(query, k=3, index_type="ivfflat")
    results_hnsw = search_pgvector(query, k=3, index_type="hnsw")
    print("IVFFlat:", json.dumps(results_ivfflat, indent=2))
    print("HNSW:", json.dumps(results_hnsw, indent=2))
finally:
    cur.close()
    conn.close()