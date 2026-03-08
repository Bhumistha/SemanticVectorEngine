import faiss
import pickle
import numpy as np

# Load embeddings
with open("data/embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

embeddings = np.array(embeddings).astype("float32")

print("Embeddings loaded:", embeddings.shape)


"""
FAISS IndexFlatL2 is used because:
- It performs exact nearest neighbor search
- Works well for medium sized datasets
- No training required
"""

dimension = embeddings.shape[1]

# Create FAISS index
index = faiss.IndexFlatL2(dimension)

# Add vectors to index
index.add(embeddings)

print("Total vectors in index:", index.ntotal)

# Save index
faiss.write_index(index, "data/faiss_index.index")

print("FAISS index saved.")