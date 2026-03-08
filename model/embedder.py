from sentence_transformers import SentenceTransformer
import pickle
import numpy as np

# Load cleaned documents
with open("data/newsgroups.pkl", "rb") as f:
    documents = pickle.load(f)

print("Documents loaded:", len(documents))


"""
Model Choice Justification:

all-MiniLM-L6-v2 is used because:
- It is lightweight
- Fast
- Produces 384 dimensional embeddings
- Good for semantic similarity
"""

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Generating embeddings...")

embeddings = model.encode(
    documents,
    show_progress_bar=True
)

print("Embedding shape:", np.array(embeddings).shape)

# Save embeddings
with open("data/embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)

print("Embeddings saved successfully.")

print("\nExample embedding:")
print(embeddings[0][:10])