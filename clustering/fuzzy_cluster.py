import pickle
import numpy as np
import skfuzzy as fuzz

# Load embeddings
with open("data/embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

embeddings = np.array(embeddings)

print("Embeddings shape:", embeddings.shape)

"""
Number of clusters chosen = 15

Reason:
The dataset has 20 labelled topics but there is semantic overlap.
Choosing slightly fewer clusters allows the fuzzy model to capture
topic mixtures more effectively.
"""

num_clusters = 15

print("Running fuzzy clustering...")

# Transpose required for scikit-fuzzy
cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
    embeddings.T,
    c=num_clusters,
    m=2,
    error=0.005,
    maxiter=1000
)

print("Clustering completed")

print("Membership matrix shape:", u.shape)

# Save cluster centers
with open("data/cluster_centers.pkl", "wb") as f:
    pickle.dump(cntr, f)

# Save membership matrix
with open("data/membership_matrix.pkl", "wb") as f:
    pickle.dump(u, f)

print("Clusters saved successfully")