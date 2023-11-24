import io
import logging
from typing import List, Union, Protocol, Dict
from pathlib import Path

from haystack.dataclasses import ByteStream
from haystack.lazy_imports import LazyImport
from haystack import Document, component, default_to_dict

with LazyImport("Run 'pip install pypdf'") as pypdf_import:
    from pypdf import PdfReader


logger = logging.getLogger(__name__)


class PyPDFConverter(Protocol):
    """
    A protocol that defines a converter which takes a PdfReader object and converts it into a Document object.
    """

    def convert(self, reader: "PdfReader") -> Document:
        ...


class DefaultConverter:
    """
    The default converter class that extracts text from a PdfReader object's pages and returns a Document.
    """

    def convert(self, reader: "PdfReader") -> Document:
        """Extract text from the PDF and return a Document object with the text content."""
        text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
        return Document(content=text)


# This registry is used to store converters names and instances.
# It can be used to register custom converters.
CONVERTERS_REGISTRY: Dict[str, PyPDFConverter] = {"default": DefaultConverter()}


@component
class PyPDFToDocument:
    """
    Converts PDF files to Document objects.
    It uses a converter that follows the PyPDFConverter protocol to perform the conversion.
    A default text extraction converter is used if no custom converter is provided.
    """

    def __init__(self, converter_name: str = "default"):
        """
        Initializes the PyPDFToDocument component with an optional custom converter.
        :param converter_name: A converter name that is registered in the CONVERTERS_REGISTRY.
            Defaults to 'default'.
        """
        pypdf_import.check()

        try:
            converter = CONVERTERS_REGISTRY[converter_name]
        except KeyError:
            msg = (
                f"Invalid converter_name: {converter_name}.\n Available converters: {list(CONVERTERS_REGISTRY.keys())}"
            )
            raise ValueError(msg) from KeyError
        self.converter_name = converter_name
        self._converter: PyPDFConverter = converter

    def to_dict(self):
        # do not serialize the _converter instance
        return default_to_dict(self, converter_name=self.converter_name)

    @component.output_types(documents=List[Document])
    def run(self, sources: List[Union[str, Path, ByteStream]]):
        """
        Converts a list of PDF sources into Document objects using the configured converter.

        :param sources: A list of PDF data sources, which can be file paths or ByteStream objects.
        :return: A dictionary containing a list of Document objects under the 'documents' key.
        """
        documents = []
        for source in sources:
            try:
                pdf_reader = self._get_pdf_reader(source)
                document = self._converter.convert(pdf_reader)
            except Exception as e:
                logger.warning("Could not read %s and convert it to Document, skipping. %s", source, e)
                continue
            documents.append(document)

        return {"documents": documents}

    def _get_pdf_reader(self, source: Union[str, Path, ByteStream]) -> "PdfReader":
        """
        Creates a PdfReader object from a given source, which can be a file path or a ByteStream object.

        :param source: The source of the PDF data.
        :return: A PdfReader instance initialized with the PDF data from the source.
        :raises ValueError: If the source type is not supported.
        """
        if isinstance(source, (str, Path)):
            return PdfReader(str(source))
        elif isinstance(source, ByteStream):
            return PdfReader(io.BytesIO(source.data))
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")
