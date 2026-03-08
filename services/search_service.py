import faiss
import pickle
import numpy as np
from config import TOP_K

index = faiss.read_index("data/faiss_index.index")

with open("data/newsgroups.pkl", "rb") as f:
    documents = pickle.load(f)

with open("data/bm25_index.pkl", "rb") as f:
    bm25 = pickle.load(f)


def hybrid_search(query, embedding):

    query_vector = embedding.reshape(1, -1)

    distances, indices = index.search(query_vector, TOP_K)

    vector_results = []

    for idx in indices[0]:

        vector_results.append({
            "doc_id": int(idx),
            "text": documents[idx][:300]
        })

    tokenized = query.split()

    bm25_scores = bm25.get_scores(tokenized)

    bm25_top = np.argsort(bm25_scores)[-TOP_K:]

    bm25_results = []

    for idx in bm25_top:

        bm25_results.append({
            "doc_id": int(idx),
            "text": documents[idx][:300]
        })

    return vector_results + bm25_results