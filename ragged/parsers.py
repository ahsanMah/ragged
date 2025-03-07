import os
import re
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

            # with open("extracted.txt", "w") as f:
            #     f.write(f"---PAGE {page_idx}---\n")
            #     f.write(text)

            for chunk_idx, chunk in enumerate(self.extract_sentences(text)):
                metadata = ChunkMetadata(
                    filename=filename,
                    file_type="pdf",
                    file_path=path,
                    page_number=page_idx,
                    chunk_index=chunk_idx,
                )
                chunk = self.post_process_chunk(chunk)
                yield (chunk, metadata)

    def post_process_chunk(self, text: str) -> str:
        """Optional post processing of the text to make it more coherent for LLM
        """

        # remove newlines to resemble paragraphs
        text = re.sub(r"[\n\s?]+", " ", text)
        return text

    def extract_sentences(self, text: str) -> Generator[str]:
        """
        Extract sentences from a text.
        """

        text = re.sub(r"[\n\s?]+", "\n", text)
        sentences = text.split(".\n")

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
