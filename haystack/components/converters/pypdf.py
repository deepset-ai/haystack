import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union

from haystack import Document, component, default_to_dict, logging
from haystack.components.converters.utils import get_bytestream_from_source, normalize_metadata
from haystack.dataclasses import ByteStream
from haystack.lazy_imports import LazyImport

with LazyImport("Run 'pip install pypdf'") as pypdf_import:
    from pypdf import PdfReader


logger = logging.getLogger(__name__)


class PyPDFConverter(Protocol):
    """
    A protocol that defines a converter which takes a PdfReader object and converts it into a Document object.
    """

    def convert(self, reader: "PdfReader", **kwargs) -> Document:
        ...


class DefaultConverter:
    """
    The default converter class that extracts text from a PdfReader object's pages and returns a Document.
    """

    def convert(self, reader: "PdfReader", **kwargs) -> Document:
        """Extract text from the PDF and return a Document object with the text content."""
        text = "\f".join(page.extract_text() for page in reader.pages)
        return Document(content=text)


class CustomConverter:
    """
    A custom converter class that extracts text from specified pages of a PdfReader object and returns a Document.
    It considers parameters like start_page and end_page.
    """

    def convert(self, reader: "PdfReader", **kwargs) -> Document:
        # Extract start_page and end_page parameters from kwargs, with default values
        start_page = kwargs.get("start_page", 0)
        end_page = kwargs.get("end_page", len(reader.pages) - 1)

        # If end_page is set to -1 or has an invalid value, process until the end of the document
        if 0 > end_page > len(reader.pages) - 1:
            end_page = len(reader.pages) - 1

        # if start_page has an invalid value, set it to 0
        if 0 > start_page > end_page:
            start_page = 0

        text = "\f".join(page.extract_text() for page in reader.pages[start_page : end_page + 1])

        return Document(content=text)


# This registry is used to store converters names and instances.
# It can be used to register custom converters.
CONVERTERS_REGISTRY: Dict[str, PyPDFConverter] = {"default": DefaultConverter(), "custom": CustomConverter()}


@component
class PyPDFToDocument:
    """
    Converts PDF files to Documents.

    Uses `pypdf` compatible converters to convert PDF files to Documents.
    A default text extraction converter is used if one is not provided.

    Usage example:
    ```python
    from haystack.components.converters.pypdf import PyPDFToDocument

    converter = PyPDFToDocument()
    results = converter.run(sources=["sample.pdf"], meta={"date_added": datetime.now().isoformat()})
    documents = results["documents"]
    print(documents[0].content)
    # 'This is a text from the PDF file.'
    ```
    """

    def __init__(self, converter_name: str = "default", conversion_params: dict = None):
        """
        Create an PyPDFToDocument component.

        :param converter_name:
            Name of the registered converter to use.
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
        self.conversion_params = conversion_params or {}

    def to_dict(self):
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        # do not serialize the _converter instance
        return default_to_dict(self, converter_name=self.converter_name, conversion_params=self.conversion_params)

    @component.output_types(documents=List[Document])
    def run(
        self,
        sources: List[Union[str, Path, ByteStream]],
        meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    ):
        """
        Converts PDF files to Documents.

        :param sources:
            List of HTML file paths or ByteStream objects.
        :param meta:
            Optional metadata to attach to the Documents.
            This value can be either a list of dictionaries or a single dictionary.
            If it's a single dictionary, its content is added to the metadata of all produced Documents.
            If it's a list, the length of the list must match the number of sources, because the two lists will be zipped.
            If `sources` contains ByteStream objects, their `meta` will be added to the output Documents.

        :returns:
            A dictionary with the following keys:
            - `documents`: Created Documents
        """
        documents = []
        meta_list = normalize_metadata(meta, sources_count=len(sources))

        for source, metadata in zip(sources, meta_list):
            try:
                bytestream = get_bytestream_from_source(source)
            except Exception as e:
                logger.warning("Could not read {source}. Skipping it. Error: {error}", source=source, error=e)
                continue
            try:
                pdf_reader = PdfReader(io.BytesIO(bytestream.data))
                document = self._converter.convert(pdf_reader, **self.conversion_params)
            except Exception as e:
                logger.warning(
                    "Could not read {source} and convert it to Document, skipping. {error}", source=source, error=e
                )
                continue

            merged_metadata = {**bytestream.meta, **metadata}
            document.meta = merged_metadata
            documents.append(document)

        return {"documents": documents}
