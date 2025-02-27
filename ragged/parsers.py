from typing import Iterable, List

from pypdf import PdfReader


class Parser:
    '''
    Parses input objects such as PDFs and HTML files.
    Each method should return a string upto a chunksize length.
    '''
    def __init__(self, chunksize: int = 256):
        self.chunksize = chunksize
        self.metadata = {}

    def parse_pdf(self, path: str) -> Iterable[str]:
        """
        Parse the text from a PDF file.
        """
        reader = PdfReader(path)
        number_of_pages = len(reader.pages)
        for i in range(number_of_pages):
            page = reader.pages[i]
            text = page.extract_text(extraction_mode="plain")

            for s in self.extract_sentences(text):
                yield s

    def extract_sentences(self, text: str) -> List[str]:
        """
        Extract sentences from a text.
        """
        sentences =  text.split("\n")

        left_end = 0
        running_size = 0
        for i in range(0, len(sentences)):
            running_size += len(sentences[i])

            if running_size < self.chunksize:
                continue

            yield ''.join(sentences[left_end : i + 1])
            left_end = i + 1
            running_size = 0

    def parse(self, path: str) -> Iterable[str]:
        """
        Parse the text from a file.
        """
        if path.endswith(".pdf"):
            iterator =  self.parse_pdf(path)
        else:
            raise ValueError("Unsupported file type")

        return iterator
