from functools import lru_cache
import numpy as np
from sentence_transformers import SentenceTransformer
from config import MODEL_NAME

model = SentenceTransformer(MODEL_NAME)


@lru_cache(maxsize=1000)
def embed_query(text: str):

    embedding = model.encode(text)

    return np.array(embedding).astype("float32")