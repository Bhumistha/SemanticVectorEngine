from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import pickle
import numpy as np
import logging
import os
import time
from functools import lru_cache
from sentence_transformers import SentenceTransformer

from cache.semantic_cache import SemanticCache
from config import TOP_K, CACHE_THRESHOLD, MODEL_NAME


# -------- Ensure Logs Folder Exists --------
os.makedirs("logs", exist_ok=True)


# -------- Logging Configuration --------
logging.basicConfig(
    filename="logs/queries.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# -------- Initialize FastAPI --------
app = FastAPI()


# -------- Load Model --------
model = SentenceTransformer(MODEL_NAME, device="cpu")


# -------- Query Embedding Cache --------
@lru_cache(maxsize=1000)
def embed_query(text: str):

    embedding = model.encode(text)

    embedding = np.array(embedding).astype("float32")

    return embedding


# -------- Load Documents --------
with open("data/newsgroups.pkl", "rb") as f:
    documents = pickle.load(f)


# -------- Load FAISS Index --------
index = faiss.read_index("data/faiss_index.index")


# -------- Load Cluster Centers --------
with open("data/cluster_centers.pkl", "rb") as f:
    cluster_centers = np.array(pickle.load(f)).astype("float32")


# -------- Initialize Semantic Cache --------
cache = SemanticCache(threshold=CACHE_THRESHOLD)


# -------- Cluster Detection --------
def get_query_cluster(query_embedding):

    distances = np.linalg.norm(
        cluster_centers - query_embedding,
        axis=1
    )

    cluster_id = np.argmin(distances)

    return int(cluster_id)


# -------- Request Schema --------
class QueryRequest(BaseModel):
    query: str


# -------- Main Query Endpoint --------
@app.post("/query")
def query_api(request: QueryRequest):

    start_time = time.time()

    query = request.query

    logging.info(f"User Query: {query}")

    try:

        # Convert query → embedding (cached)
        query_embedding = embed_query(query)

        # Detect cluster
        query_cluster = get_query_cluster(query_embedding)

        # -------- Check Cache --------
        cached, similarity = cache.lookup(query_embedding, query_cluster)

        if cached:

            logging.info(f"Cache HIT | Query: {query}")

            latency = round((time.time() - start_time) * 1000, 2)

            return {
                "query": query,
                "cache_hit": True,
                "latency_ms": latency,
                "matched_query": cached["query"],
                "similarity_score": float(similarity),
                "dominant_cluster": cached["cluster"],
                "results": cached["result"]
            }

        logging.info(f"Cache MISS | Query: {query}")

        # -------- FAISS Search --------
        query_vector = query_embedding.reshape(1, -1)

        distances, indices = index.search(query_vector, TOP_K)

        results = []

        for score, idx in zip(distances[0], indices[0]):

            if idx == -1:
                continue

            results.append({
                "doc_id": int(idx),
                "score": round(float(score), 4),
                "text": documents[idx][:300]
            })

        # -------- Store in Cache --------
        cache.add(query, query_embedding, results, query_cluster)

        latency = round((time.time() - start_time) * 1000, 2)

        return {
            "query": query,
            "cache_hit": False,
            "dominant_cluster": query_cluster,
            "result_count": len(results),
            "latency_ms": latency,
            "results": results
        }

    except Exception as e:

        logging.error(str(e))

        return {"error": str(e)}


# -------- Cache Stats Endpoint --------
@app.get("/cache/stats")
def cache_stats():

    return cache.stats()


# -------- Clear Cache Endpoint --------
@app.delete("/cache")
def clear_cache():

    cache.clear()

    return {
        "message": "Cache cleared successfully"
    }


# -------- Health Endpoint --------
@app.get("/health")
def health():

    return {
        "status": "ok",
        "service": "semantic-search-api"
    }