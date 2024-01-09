import io
import logging
from typing import List, Union, Protocol, Dict, Any, Optional
from pathlib import Path

from haystack.dataclasses import ByteStream
from haystack.lazy_imports import LazyImport
from haystack import Document, component, default_to_dict
from haystack.components.converters.utils import get_bytestream_from_source, normalize_metadata

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
    def run(
        self,
        sources: List[Union[str, Path, ByteStream]],
        meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    ):
        """
        Converts a list of PDF sources into Document objects using the configured converter.

        :param sources: A list of PDF data sources, which can be file paths or ByteStream objects.
        :param meta: Optional metadata to attach to the Documents.
          This value can be either a list of dictionaries or a single dictionary.
          If it's a single dictionary, its content is added to the metadata of all produced Documents.
          If it's a list, the length of the list must match the number of sources, because the two lists will be zipped.
          Defaults to `None`.
        :return: A dictionary containing a list of Document objects under the 'documents' key.
        """
        documents = []
        meta_list = normalize_metadata(meta, sources_count=len(sources))

        for source, metadata in zip(sources, meta_list):
            try:
                bytestream = get_bytestream_from_source(source)
            except Exception as e:
                logger.warning("Could not read %s. Skipping it. Error: %s", source, e)
                continue
            try:
                pdf_reader = PdfReader(io.BytesIO(bytestream.data))
                document = self._converter.convert(pdf_reader)
            except Exception as e:
                logger.warning("Could not read %s and convert it to Document, skipping. %s", source, e)
                continue

            merged_metadata = {**bytestream.meta, **metadata}
            document.meta = merged_metadata
            documents.append(document)

        return {"documents": documents}
