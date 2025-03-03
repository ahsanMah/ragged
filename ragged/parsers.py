import os
from typing import Generator, Iterable, List

from pypdf import PdfReader

from ragged.documents import ChunkMetadata


class Parser:
    """
    Parses input objects such as PDFs and HTML files.
    Each method should return a string upto a chunksize length.
    """

    def __init__(self, chunksize: int = 256):
        self.chunksize = chunksize
        self.metadata = {}

    def parse_pdf(self, path: str) -> Iterable[str]:
        """
        Parse the text from a PDF file.
        """
        filename = os.path.basename(path)
        reader = PdfReader(path)
        number_of_pages = len(reader.pages)
        for page_idx in range(number_of_pages):
            page = reader.pages[page_idx]
            text = page.extract_text(extraction_mode="plain")

            for chunk_idx, chunk in enumerate(self.extract_sentences(text)):
                metadata = ChunkMetadata(
                    filename=filename,
                    file_type="pdf",
                    file_path=path,
                    page_number=page_idx,
                    chunk_index=chunk_idx,
                )
                yield (chunk, metadata)

    def extract_sentences(self, text: str) -> Generator[str]:
        """
        Extract sentences from a text.
        """
        sentences = text.split("\n")

        start = 0
        running_size = 0
        for end in range(0, len(sentences)):
            running_size += len(sentences[end])

            if running_size < self.chunksize:
                continue

            yield "".join(sentences[start : end + 1])
            start = end + 1
            running_size = 0

    def parse(self, path: str) -> Iterable[str]:
        """
        Parse the text from a file.
        """
        if path.endswith(".pdf"):
            iterator = self.parse_pdf(path)
        else:
            raise ValueError("Unsupported file type")

        return iterator
