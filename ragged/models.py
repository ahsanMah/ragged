import os
from functools import lru_cache
from typing import Optional

import torch
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer

from config import Config


class ModelManager:
    """
    Singleton class to manage model loading and persistence.
    Uses lazy loading and caching to optimize resource usage.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._llm = None
            cls._instance._embedder = None
        return cls._instance

    @property
    @lru_cache(maxsize=1)
    def llm(self) -> Llama:
        """
        Lazily load and cache the LLM model.
        Uses lru_cache to ensure the model is only loaded once.
        """
        if self._llm is None:
            self._llm = Llama.from_pretrained(
                repo_id="bartowski/microsoft_Phi-4-mini-instruct-GGUF",
                filename="*Q4_K_M.gguf",
                n_gpu_layers=20,
                n_ctx=32768,
                verbose=True,
                local_dir=Config.MODEL_DIR,
            )
        return self._llm

    @property
    @lru_cache(maxsize=1)
    def embedder(self) -> SentenceTransformer:
        """
        Lazily load and cache the embedding model.
        Uses lru_cache to ensure the model is only loaded once.
        """
        if self._embedder is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._embedder = SentenceTransformer(
                "NovaSearch/stella_en_400M_v5",
                trust_remote_code=True,
                cache_folder=Config.EMBEDDING_MODEL_DIR,
                device=device,
                config_kwargs={
                    "use_memory_efficient_attention": device.type == "cuda",
                    "unpad_inputs": device.type == "cuda",
                },
                default_prompt_name="s2p_query",
            )
        return self._embedder

    def close(self):
        """
        Properly close and clean up model resources.
        """
        if self._llm is not None:
            self._llm.close()
            self._llm = None

        if self._embedder is not None:
            del self._embedder
            self._embedder = None
            torch.cuda.empty_cache()


# Global instance that can be imported and used throughout the application
model_manager = ModelManager()
