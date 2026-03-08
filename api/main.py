from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import pickle
import numpy as np
import logging
import os
import time
import threading
from functools import lru_cache
from sentence_transformers import SentenceTransformer

from cache.semantic_cache import SemanticCache
from config import TOP_K, CACHE_THRESHOLD, MODEL_NAME


print("Starting FastAPI service...")

# -------- Ensure Logs Folder --------
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/queries.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

app = FastAPI()


# -------- Global Variables --------
model = None
documents = None
index = None
cluster_centers = None
cache = None
ready = False


# -------- Background Loader --------
def load_resources():
    global model, documents, index, cluster_centers, cache, ready

    try:
        print("Loading model and FAISS index...")

        model = SentenceTransformer(MODEL_NAME)

        with open("data/newsgroups.pkl", "rb") as f:
            documents = pickle.load(f)

        index = faiss.read_index("data/faiss_index.index")

        with open("data/cluster_centers.pkl", "rb") as f:
            cluster_centers = pickle.load(f)

        cache = SemanticCache(threshold=CACHE_THRESHOLD)

        ready = True

        print("Resources loaded successfully")

    except Exception as e:
        print("Startup error:", str(e))


@app.on_event("startup")
def startup():
    thread = threading.Thread(target=load_resources)
    thread.start()


# -------- Root Endpoint --------
@app.get("/")
def root():
    return {"status": "API running", "ready": ready}


# -------- Health Check --------
@app.get("/health")
def health():
    return {"ready": ready}


@lru_cache(maxsize=1000)
def embed_query(text: str):
    embedding = model.encode(text)
    return np.array(embedding).astype("float32")


def get_query_cluster(query_embedding):
    distances = np.linalg.norm(cluster_centers - query_embedding, axis=1)
    return int(np.argmin(distances))


class QueryRequest(BaseModel):
    query: str


# -------- Main Query Endpoint --------
@app.post("/query")
def query_api(request: QueryRequest):

    if not ready:
        return {"message": "Model loading, try again shortly"}

    start_time = time.time()
    query = request.query

    logging.info(f"Query: {query}")

    query_embedding = embed_query(query)

    query_cluster = get_query_cluster(query_embedding)

    cached, similarity = cache.lookup(query_embedding, query_cluster)

    if cached:
        latency = round((time.time() - start_time) * 1000, 2)

        return {
            "query": query,
            "cache_hit": True,
            "latency_ms": latency,
            "results": cached["result"]
        }

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

    cache.add(query, query_embedding, results, query_cluster)

    latency = round((time.time() - start_time) * 1000, 2)

    return {
        "query": query,
        "cache_hit": False,
        "latency_ms": latency,
        "results": results
    }