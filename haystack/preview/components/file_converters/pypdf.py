import io
import logging
from typing import List, Union, Callable, Optional
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
    Converts a PDF file to a Document. By default, it extracts the text from the PDF file and creates a Document
    instance with the extracted text. You can also pass a custom converter function to the component.
    """

    def __init__(self, converter: Optional[Callable[[PdfReader], Document]] = None):
        """
        Initializes the PyPDFToDocument component with an optional custom converter.
        """
        pypdf_import.check()
        if converter and not callable(converter):
            raise ValueError("Converter must be a callable accepting a PdfReader object and returning a Document")
        self.converter = (
            converter
            if converter
            else lambda pdf_reader: Document(
                content="".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
            )
        )

    @component.output_types(documents=List[Document])
    def run(self, sources: List[Union[str, Path, ByteStream]]):
        """
        Converts PDF files to Documents.

        :param sources: A list of PDF data sources
        """
        documents = []
        for source in sources:
            try:
                pdf_reader = self._get_pdf_reader(source)
                document = self.converter(pdf_reader)
            except Exception as e:
                logger.warning("Could not read %s and convert it to Document, skipping. %s", source, e)
                continue
            documents.append(document)

        return {"documents": documents}

    def _get_pdf_reader(self, source: Union[str, Path, ByteStream]) -> PdfReader:
        """
        Creates a PdfReader object from the given source.

        :param source: PDF file data source
        :return: PdfReader object
        """
        if isinstance(source, (str, Path)):
            return PdfReader(str(source))
        elif isinstance(source, ByteStream):
            return PdfReader(io.BytesIO(source.data))
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")
