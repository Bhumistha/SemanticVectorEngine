import pickle
import numpy as np

with open("data/membership_matrix.pkl", "rb") as f:
    membership = pickle.load(f)

membership = np.array(membership)

doc_id = 100

print("Cluster distribution for document:", doc_id)

for cluster_id, value in enumerate(membership[:, doc_id]):
    print(f"Cluster {cluster_id}: {value:.3f}")