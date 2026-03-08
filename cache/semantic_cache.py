from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, OrderedDict


class SemanticCache:

    def __init__(self, threshold=0.85, max_size=500):

        # cache is grouped by cluster
        self.cache = defaultdict(OrderedDict)

        self.threshold = threshold
        self.max_size = max_size

    # -------- Lookup Query in Cache --------
    def lookup(self, embedding, cluster):

        cluster_entries = self.cache[cluster]

        for key, entry in cluster_entries.items():

            sim = cosine_similarity(
                [embedding],
                [entry["embedding"]]
            )[0][0]

            # if similarity above threshold → cache hit
            if sim >= self.threshold:

                # move to end (LRU behavior)
                cluster_entries.move_to_end(key)

                return entry, sim

        return None, None

    # -------- Add Query to Cache --------
    def add(self, query, embedding, result, cluster):

        cluster_entries = self.cache[cluster]

        # LRU eviction
        if len(cluster_entries) >= self.max_size:
            cluster_entries.popitem(last=False)

        cluster_entries[query] = {
            "query": query,
            "embedding": embedding,
            "result": result,
            "cluster": cluster
        }

    # -------- Cache Statistics --------
    def stats(self):

        total_entries = sum(len(v) for v in self.cache.values())

        return {
            "total_cache_entries": total_entries,
            "clusters": len(self.cache)
        }

    # -------- Clear Cache --------
    def clear(self):

        self.cache.clear()