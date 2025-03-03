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
