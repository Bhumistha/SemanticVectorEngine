import numpy as np
import pickle

with open("data/cluster_centers.pkl", "rb") as f:
    cluster_centers = pickle.load(f)


def get_query_cluster(embedding):

    distances = np.linalg.norm(
        cluster_centers - embedding,
        axis=1
    )

    return int(np.argmin(distances))