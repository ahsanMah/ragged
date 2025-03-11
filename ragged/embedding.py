from typing import List, Literal, Union

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from ragged.models import model_manager


class Embedder:
    def __init__(
        self,
    ):
        """
        Initialize the embedder. The model will be accessed synchronously
        after the ModelManager has been initialized.
        """
        self.model = model_manager.get_embedder()

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

    def k_nearest_neighbors(
        self,
        embeddings: np.ndarray,
        query: str,
        k: int = 20,
        score_threshold: float = 0.4,
    ):
        """
        Find k nearest neighbors for a query in the embedding space.
        """
        knn = NearestNeighbors(n_neighbors=min(k, len(embeddings)), metric="cosine")
        knn.fit(embeddings)
        query_embedding = self.model.encode([query])
        scores, indices = knn.kneighbors(query_embedding, return_distance=True)
        scores = scores.flatten()
        indices = indices.flatten()

        # Filter out scores below the threshold
        scores = scores[scores >= score_threshold]
        indices = indices[scores >= score_threshold]

        return scores, indices
