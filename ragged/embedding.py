from typing import List, Literal, Union

import torch
from sklearn.neighbors import NearestNeighbors

from ragged.models import model_manager


class Embedder:
    def __init__(
        self,
    ):
        self.model = model_manager.embedder

    def embed(self, data: List[str], batch_size: int = 16):
        """
        Embeds the data into a vector space using a predefined model.
        """
        embeddings = self.model.encode(
            data, batch_size=batch_size, show_progress_bar=True
        )
        return embeddings

    def embed_and_persist(self, data: List[str], path: str):
        """
        Embeds data and saves the embeddings to disk.
        """
        pass

    def k_nearest_neighbors(self, embeddings, query: str, k: int = 5):
        """
        Find k nearest neighbors for a query in the embedding space.
        """
        knn = NearestNeighbors(n_neighbors=5, metric="cosine")
        knn.fit(embeddings)
        query_embedding = self.model.encode([query])
        indices = knn.kneighbors(query_embedding, return_distance=False)
        return indices.flatten()
