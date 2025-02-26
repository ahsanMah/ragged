from typing import List, Literal, Union

import torch
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

from config import Config


class Embedder:

    def __init__(
        self,
        device: Union[str, torch.device],
        model_dir: str = Config.EMBEDDING_MODEL_DIR,
        prompt_name: Literal["s2p_query", "s2s_query"] = "s2s_query",
    ):
        device = torch.device(device)
        self.model = SentenceTransformer(
            "NovaSearch/stella_en_400M_v5",
            trust_remote_code=True,
            cache_folder=model_dir,
            device=device,
            config_kwargs={
                "use_memory_efficient_attention": device.type == "cuda",
                "unpad_inputs": device.type == "cuda",
            },
            default_prompt_name=prompt_name,
        )

    def embed(self, data: List[str]):
        """
        Embeds the data into a vector space using a predefined model.
        """
        embeddings = self.model.encode(data, batch_size=16, show_progress_bar=True)
        return embeddings

    def embed_and_persist(self, data: List[str], path: str):
        """"""
        pass

    def k_nearest_neighbors(self, embeddings, query: str, k: int = 5):
        """"""
        knn = NearestNeighbors(n_neighbors=5, metric="cosine")
        knn.fit(embeddings)
        query_embedding = self.model.encode([query])
        indices = knn.kneighbors(query_embedding, return_distance=False)
        return indices.flatten()
