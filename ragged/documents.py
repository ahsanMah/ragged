import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ChunkMetadata:
    """
    Struct-like object which stores metadata for a text chunk extracted from a document
    """

    # Source document information
    filename: str
    file_path: str
    file_type: str  # pdf, txt, etc.

    # Location within document
    page_number: int
    chunk_index: int

    # Content boundaries
    start_char: Optional[int] = None
    end_char: Optional[int] = None

    # Additional context
    section_title: Optional[str] = None
    preceding_heading: Optional[str] = None

    def to_dict(self):
        """Convert metadata to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        """Create metadata instance from dictionary."""
        return cls(**data)


class Document:
    """
    Stores document chunks, embeddings, and metadata.
    """

    def __init__(self, document_name: str, storage_dir: str = "/tmp/ragged"):
        self.name = document_name
        self.storage_dir = os.path.join(storage_dir, self.name)
        os.makedirs(storage_dir, exist_ok=True)

        # File paths for storage
        self.embeddings_path = os.path.join(storage_dir, "embeddings.npy")
        self.chunks_path = os.path.join(storage_dir, "textchunks.npy")
        self.metadata_path = os.path.join(storage_dir, "metadata.json")

        # In-memory cache
        self.embeddings = None
        self.chunks = None
        self.metadata = None

    def store(
        self, chunks: List[str], embeddings: np.ndarray, metadata: List[ChunkMetadata]
    ):
        """
        Store document chunks, embeddings and metadata.

        Args:
            chunks: List of text chunks
            embeddings: Numpy array of embeddings
            metadata: List of chunk metadata
        """
        # Convert metadata to serializable format
        serialized_metadata = [m.to_dict() for m in metadata]

        # Save data to disk
        np.save(self.embeddings_path, embeddings, allow_pickle=False)
        np.save(self.chunks_path, np.array(chunks, dtype=object), allow_pickle=True)

        with open(self.metadata_path, "w") as f:
            json.dump(serialized_metadata, f)

        # Update in-memory cache
        self.embeddings = embeddings
        self.chunks = chunks
        self.metadata = metadata

    def load(self) -> Tuple[np.ndarray, List[str], List[ChunkMetadata]]:
        """
        Load stored embeddings, chunks and metadata.

        Returns:
            Tuple of (embeddings, chunks, metadata)
        """
        if (
            os.path.exists(self.embeddings_path)
            and os.path.exists(self.chunks_path)
            and os.path.exists(self.metadata_path)
        ):
            # Load data from disk
            embeddings = np.load(self.embeddings_path, allow_pickle=False)
            chunks = np.load(self.chunks_path, allow_pickle=True).tolist()

            with open(self.metadata_path, "r") as f:
                serialized_metadata = json.load(f)

            # Deserialize metadata
            metadata = [ChunkMetadata.from_dict(m) for m in serialized_metadata]

            # Update in-memory cache
            # self.embeddings = embeddings
            # self.chunks = chunks
            # self.metadata = metadata

            return embeddings, chunks, metadata
        else:
            raise FileNotFoundError("Store files not found. Process document first.")
