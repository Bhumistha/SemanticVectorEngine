import pickle
import numpy as np
import umap
import plotly.express as px

# Load embeddings
with open("data/embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

embeddings = np.array(embeddings)

# Load membership matrix
with open("data/membership_matrix.pkl", "rb") as f:
    membership = pickle.load(f)

membership = np.array(membership)

# Get dominant cluster per document
clusters = np.argmax(membership, axis=0)

print("Running UMAP dimensionality reduction...")

reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    metric="cosine"
)

embedding_2d = reducer.fit_transform(embeddings)

print("UMAP complete")

# Create visualization
fig = px.scatter(
    x=embedding_2d[:,0],
    y=embedding_2d[:,1],
    color=clusters.astype(str),
    title="Document Cluster Map (UMAP)",
)

fig.show()