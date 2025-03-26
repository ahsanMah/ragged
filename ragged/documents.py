import hashlib
import json
import os
import pdb
import sqlite3
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

    # # Content boundaries
    # start_char: Optional[int] = None
    # end_char: Optional[int] = None

    # # Additional context
    # section_title: Optional[str] = None
    # preceding_heading: Optional[str] = None

    def to_dict(self):
        """Convert metadata to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        """Create metadata instance from dictionary."""
        return cls(**data)


class DocumentDB:
    """
    Stores document chunks, embeddings, and metadata.
    """

    @staticmethod
    def get_name_from_path(p):
        return os.path.basename(p).split(".")[0]

    def __init__(
        self, project_name: str, storage_dir: str = "/tmp/ragged", persist: int = False
    ):
        self.name = project_name
        self.storage_dir = storage_dir
        self.persist = persist
        if persist:
            os.makedirs(storage_dir, exist_ok=True)
            db = os.path.join(storage_dir, f"{self.name}.db")
        else:
            db = "file:{self.name}?mode=memory&cache=shared"
        self.database_path = db
        self.con = sqlite3.connect(self.database_path)

        # Using hash of text-chunk as unique ID
        # Have separate tables for chunk+metadata and vectors
        self.con.execute(
            """CREATE TABLE IF NOT EXISTS vectors(
            hash BLOB,
            embedding BLOB)
            """
        )
        self.con.execute(
            """CREATE TABLE IF NOT EXISTS metadata(
            hash BLOB,
            text_chunk TEXT,
            filename TEXT,
            file_path TEXT,
            file_type TEXT,
            page_number INTEGER,
            chunk_index INTEGER)
            """
        )

        self.metadata_path = os.path.join(self.storage_dir, "metadata.json")

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

        hashes = [hash_text_binary(c) for c in chunks]
        vector_bytes = [v.tobytes() for v in embeddings]
        rows = list(zip(hashes, vector_bytes))

        # rows = list(zip(chunks, vector_bytes, [self.name] * len(chunks)))
        sql_placeholders = ", ".join(["?" for _ in range(len(rows[0]))])

        # Save data to database
        with self.con:
            rows = list(zip(hashes, vector_bytes))

            # rows = list(zip(chunks, vector_bytes, [self.name] * len(chunks)))
            sql_placeholders = ", ".join(["?" for _ in range(len(rows[0]))])

            self.con.executemany(
                f"INSERT INTO vectors (hash, embedding) VALUES({sql_placeholders})",
                rows,
            )

            rows = []
            for h, c, m in zip(hashes, chunks, metadata):
                d = {"hash": h, "text_chunk": c}
                d.update(m.to_dict())
                rows.append(d)
            sql_placeholders = ", ".join(f":{k}" for k in d.keys())
            self.con.executemany(
                f"INSERT INTO metadata VALUES({sql_placeholders})",
                rows,
            )

        # Convert metadata to serializable format
        serialized_metadata = [m.to_dict() for m in metadata]
        with open(self.metadata_path, "a") as f:
            json.dump(serialized_metadata, f)

        # Update in-memory cache
        self.embeddings = embeddings
        self.chunks = chunks
        self.metadata = metadata

    def load_all(self) -> Tuple[np.ndarray, List[str], List[ChunkMetadata]]:
        """
        Load stored embeddings, chunks and metadata.

        Returns:
            Tuple of (embeddings, chunks, metadata)
        """

        if self.persist and not os.path.exists(self.storage_dir):
                raise FileNotFoundError("Store files not found. Process document first.")

        with self.con:
            rows = self.con.execute("SELECT * FROM vectors").fetchall()
            assert len(rows) > 0, "No rows found in database. Process document first."

            ids, vector_bytes = list(zip(*rows))
            embeddings = convert_bytes_to_vectors(vector_bytes)

            rows = self.con.execute("SELECT * FROM metadata").fetchall()
            ids, text_chunks, *metadata_cols = list(zip(*rows))
            metadata_rows = list(zip(*metadata_cols))
            metadata = [ChunkMetadata(*cols) for cols in metadata_rows]

        return embeddings, text_chunks, metadata


def convert_bytes_to_vectors(vector_bytes):
    N = len(vector_bytes)
    vec = np.frombuffer(bytearray(b"".join(vector_bytes)), dtype=np.float32)

    return vec.reshape((N, 1024))


def hash_text_binary(text):
    return hashlib.sha256(text.encode()).digest()  # 32-byte binary hash
