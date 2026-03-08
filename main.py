from sklearn.datasets import fetch_20newsgroups
import re
import pickle

print("Loading dataset... please wait")

"""
Headers, footers and quotes are removed because they often
contain metadata like email addresses and reply chains
that leak topic information without representing
the true semantic content of the document.
"""

dataset = fetch_20newsgroups(
    subset='all',
    remove=('headers', 'footers', 'quotes')
)

documents = dataset.data
labels = dataset.target

print("Total documents:", len(documents))

print("\nExample document:\n")
print(documents[0][:500])


def clean_text(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


documents = [clean_text(doc) for doc in documents]

print("\nCleaned example:\n")
print(documents[0][:500])


# Save dataset
with open("data/newsgroups.pkl", "wb") as f:
    pickle.dump(documents, f)