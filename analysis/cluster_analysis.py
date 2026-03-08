import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Load documents
with open("data/newsgroups.pkl", "rb") as f:
    documents = pickle.load(f)

# Load embeddings
with open("data/embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

# Load cluster centers
with open("data/cluster_centers.pkl", "rb") as f:
    centers = pickle.load(f)

# Load membership matrix
with open("data/membership_matrix.pkl", "rb") as f:
    membership = pickle.load(f)

embeddings = np.array(embeddings)
membership = np.array(membership)

num_clusters = centers.shape[0]

print("Total clusters:", num_clusters)
print("Total documents:", len(documents))


# -----------------------------
# STEP 1 — Top Documents per Cluster
# -----------------------------

print("\n==============================")
print("Top Documents Per Cluster")
print("==============================")

for cluster_id in range(num_clusters):

    print(f"\nCluster {cluster_id}")
    print("---------------------")

    scores = membership[cluster_id]

    top_docs = np.argsort(scores)[-3:][::-1]

    for doc_id in top_docs:
        print(f"Score: {scores[doc_id]:.3f}")
        print(documents[doc_id][:200])
        print()


# -----------------------------
# STEP 2 — Ambiguous Documents
# -----------------------------

print("\n==============================")
print("Ambiguous Documents")
print("==============================")

for doc_id in range(100):

    probs = membership[:, doc_id]

    sorted_probs = np.sort(probs)[::-1]

    if sorted_probs[0] - sorted_probs[1] < 0.05:

        print("\nDocument ID:", doc_id)
        print("Cluster probabilities:", sorted_probs[:3])
        print(documents[doc_id][:200])


# -----------------------------
# STEP 3 — Boundary Documents
# -----------------------------

print("\n==============================")
print("Boundary Documents")
print("==============================")

for doc_id in range(100):

    probs = membership[:, doc_id]

    max_prob = np.max(probs)

    if max_prob < 0.40:

        print("\nDocument ID:", doc_id)
        print("Max membership:", max_prob)
        print(documents[doc_id][:200])


# -----------------------------
# STEP 4 — Automatic Cluster Keywords
# -----------------------------

print("\n==============================")
print("Cluster Keywords")
print("==============================")

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

tfidf_matrix = vectorizer.fit_transform(documents)

feature_names = vectorizer.get_feature_names_out()

for cluster_id in range(num_clusters):

    scores = membership[cluster_id]

    top_docs = np.argsort(scores)[-50:]

    cluster_text = [documents[i] for i in top_docs]

    tfidf_cluster = vectorizer.transform(cluster_text)

    word_scores = np.array(tfidf_cluster.mean(axis=0)).flatten()

    top_words = word_scores.argsort()[-10:][::-1]

    keywords = [feature_names[i] for i in top_words]

    print(f"\nCluster {cluster_id} Keywords:")
    print(", ".join(keywords))