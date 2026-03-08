import pickle
import numpy as np
import umap

print("Loading embeddings...")

# Load embeddings
with open("data/embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

embeddings = np.array(embeddings)

print("Embeddings shape:", embeddings.shape)

print("Running UMAP dimensionality reduction...")
print("This may take around 30–60 seconds depending on dataset size.")

reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    metric="cosine",
    random_state=42
)

embedding_2d = reducer.fit_transform(embeddings)

print("UMAP completed.")

print("Saving reduced embeddings...")

with open("data/umap_embeddings.pkl", "wb") as f:
    pickle.dump(embedding_2d, f)

print("Saved successfully to data/umap_embeddings.pkl")
print("You can now load clusters instantly in Streamlit.")