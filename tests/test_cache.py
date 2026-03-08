from semantic_cache import SemanticCache
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

cache = SemanticCache(threshold=0.85)

query1 = "space missions"
query2 = "satellite launches"

emb1 = model.encode(query1)
emb2 = model.encode(query2)

# First query → cache miss
entry, sim = cache.lookup(emb1)

if entry is None:
    print("Cache miss")
    cache.add(query1, emb1, "result for space missions", 3)

# Second query → likely cache hit
entry, sim = cache.lookup(emb2)

if entry:
    print("Cache hit!")
    print("Matched query:", entry["query"])
    print("Similarity:", sim)