import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Load FAISS index
index = faiss.read_index("data/faiss_index.index")

# Load documents
with open("data/newsgroups.pkl", "rb") as f:
    documents = pickle.load(f)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

query = input("Enter your search query: ")

query_embedding = model.encode([query])
query_embedding = np.array(query_embedding).astype("float32")

k = 5

distances, indices = index.search(query_embedding, k)

print("\nTop results:\n")

for i in indices[0]:
    print(documents[i][:300])
    print("\n-------------------\n")