from functools import lru_cache

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL,
)


class E5Embeddings(Embeddings):
    """
    Wrapper for intfloat/multilingual-e5-large.

    E5 requires:
      - query: ... for queries
      - passage: ... for indexed document chunks
    """

    def __init__(self):
        self.model = SentenceTransformer(
            EMBEDDING_MODEL,
            device=EMBEDDING_DEVICE,
        )

    def embed_documents(self, texts):
        prepared = [f"passage: {text}" for text in texts]
        vectors = self.model.encode(
            prepared,
            batch_size=EMBEDDING_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, text):
        vector = self.model.encode(
            [f"query: {text}"],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return vector.tolist()


@lru_cache(maxsize=1)
def get_embeddings():
    return E5Embeddings()
