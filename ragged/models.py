import asyncio
from concurrent.futures import ThreadPoolExecutor

import torch
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer

from config import Config


class ModelManager:
    """
    Singleton class to manage model loading and persistence.
    Uses lazy loading and caching to optimize resource usage.
    Supports both async initialization and sync access after initialization.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._llm = None
            cls._instance._embedder = None
            cls._instance._initialized = False
            cls._instance._initialization_lock = asyncio.Lock()
        return cls._instance

    async def initialize(self):
        """
        Asynchronously initialize all models.
        This method should be called at application startup to preload models.
        """
        if self._initialized:
            return

        async with self._initialization_lock:
            if self._initialized:  # Double check in case another task initialized while waiting
                return

            # Create tasks for loading models
            llm_task = asyncio.create_task(self._init_llm())
            embedder_task = asyncio.create_task(self._init_embedder())

            # Wait for both models to load
            await asyncio.gather(llm_task, embedder_task)
            self._initialized = True

    async def _init_llm(self) -> None:
        """
        Asynchronously initialize the LLM model.
        """
        if self._llm is not None:
            return

        # Run CPU-intensive model loading in a thread
        self._llm = await asyncio.get_event_loop().run_in_executor(
            None, # using default executor
            lambda: Llama.from_pretrained(
                repo_id="bartowski/microsoft_Phi-4-mini-instruct-GGUF",
                filename="*Q4_K_M.gguf",
                n_gpu_layers=20,
                n_ctx=32768,
                verbose=True,
                local_dir=Config.MODEL_DIR,
            )
        )

    async def _init_embedder(self) -> None:
        """
        Asynchronously initialize the embedding model.
        """
        if self._embedder is not None:
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Run CPU-intensive model loading in a thread
        self._embedder = await asyncio.get_event_loop().run_in_executor(
            None, # using default executor
            lambda: SentenceTransformer(
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
        )

    def get_llm(self) -> Llama:
        """
        Synchronously get the LLM model.
        Should only be called after initialize() has completed.
        Raises RuntimeError if models aren't initialized.
        """
        if not self._initialized or self._llm is None:
            raise RuntimeError("Models not initialized. Call initialize() first.")
        return self._llm

    def get_embedder(self) -> SentenceTransformer:
        """
        Synchronously get the embedding model.
        Should only be called after initialize() has completed.
        Raises RuntimeError if models aren't initialized.
        """
        if not self._initialized or self._embedder is None:
            raise RuntimeError("Models not initialized. Call initialize() first.")
        return self._embedder

    @property
    async def llm(self) -> Llama:
        """
        Asynchronously get the LLM model.
        Initializes the model if not already loaded.
        """
        if not self._initialized or self._llm is None:
            await self._init_llm()
        return self._llm

    @property
    async def embedder(self) -> SentenceTransformer:
        """
        Asynchronously get the embedding model.
        Initializes the model if not already loaded.
        """
        if not self._initialized or self._embedder is None:
            await self._init_embedder()
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
        
        self._initialized = False
        # self._executor.shutdown(wait=True)

# Global instance that can be imported and used throughout the application
model_manager = ModelManager()
