import io
import logging
from typing import List, Union
from pathlib import Path

from haystack.preview.dataclasses import ByteStream
from haystack.preview.lazy_imports import LazyImport
from haystack.preview import Document, component

with LazyImport("Run 'pip install pypdf'") as pypdf_import:
    from pypdf import PdfReader


logger = logging.getLogger(__name__)


@component
class PyPDFToDocument:
    """
    Converts a PDF file to a Document.
    """

    def __init__(self):
        """
        Initializes the PyPDFToDocument component.
        """
        pypdf_import.check()

    @component.output_types(documents=List[Document])
    def run(self, sources: List[Union[str, Path, ByteStream]]):
        """
        Converts PDF files to Documents.

        :param sources: A list of PDF data sources
        """
        documents = []
        for source in sources:
            try:
                text = self._read_pdf_file(source)
            except Exception as e:
                logger.warning("Could not read %s. Skipping it. Error message: %s", source, e)
                continue

            document = Document(text=text)
            documents.append(document)

        return {"documents": documents}

    def _read_pdf_file(self, source: Union[str, Path, ByteStream]) -> str:
        """
        Extracts content from the given PDF source.
        :param source:  PDF file data source
        :return: The extracted text.
        """
        if isinstance(source, (str, Path)):
            pdf_reader = PdfReader(str(source))
        elif isinstance(source, ByteStream):
            pdf_reader = PdfReader(io.BytesIO(source.data))
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")

        text = "".join(extracted_text for page in pdf_reader.pages if (extracted_text := page.extract_text()))

        return text
