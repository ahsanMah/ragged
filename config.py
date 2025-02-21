import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    MODEL_DIR: str = os.getenv("MODEL_DIR", "./models")
    EMBEDDING_MODEL_DIR: str = os.getenv("EMBEDDING_MODEL_DIR", "./models")
    EMBEDDING_CACHE_DIR: str = os.getenv("EMBEDDING_MODEL_DIR", "./embedding_cache")
