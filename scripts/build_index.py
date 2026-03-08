from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

docs = [
    "sample document one",
    "sample document two"
]

embeddings = model.encode(docs)

dim = embeddings.shape[1]

index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings))

faiss.write_index(index, "data/faiss_index.index")

print("Index built successfully")