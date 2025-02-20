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
            #TODO: Add a check for line endings
            # chunks should not split sentences ideally
            for j in range(0, len(text), self.chunksize):
                yield text[j:j + self.chunksize]

    def parse(self, path: str) -> Iterable[str]:
        """
        Parse the text from a file.
        """
        if path.endswith(".pdf"):
            iterator =  self.parse_pdf(path)
        else:
            raise ValueError("Unsupported file type")

        return iterator
