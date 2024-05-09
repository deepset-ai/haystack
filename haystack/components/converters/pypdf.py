# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import io
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union

from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.components.converters.utils import get_bytestream_from_source, normalize_metadata
from haystack.dataclasses import ByteStream
from haystack.lazy_imports import LazyImport
from haystack.utils.type_serialization import deserialize_type

with LazyImport("Run 'pip install pypdf'") as pypdf_import:
    from pypdf import PdfReader


logger = logging.getLogger(__name__)


class PyPDFConverter(Protocol):
    """
    A protocol that defines a converter which takes a PdfReader object and converts it into a Document object.
    """

    def convert(self, reader: "PdfReader") -> Document:  # noqa: D102
        ...

    def to_dict(self):  # noqa: D102
        ...

    @classmethod
    def from_dict(cls, data):  # noqa: D102
        ...


class DefaultConverter:
    """
    The default converter class that extracts text from a PdfReader object's pages and returns a Document.
    """

    def convert(self, reader: "PdfReader") -> Document:
        """Extract text from the PDF and return a Document object with the text content."""
        text = "\f".join(page.extract_text() for page in reader.pages)
        return Document(content=text)

    def to_dict(self):
        """Serialize the converter to a dictionary."""
        return default_to_dict(self)

    @classmethod
    def from_dict(cls, data):
        """Deserialize the converter from a dictionary."""
        return default_from_dict(cls, data)


# This registry is used to store converters names and instances.
# It can be used to register custom converters.
# It is now deprecated and will be removed in Haystack 2.3.0.
CONVERTERS_REGISTRY: Dict[str, PyPDFConverter] = {"default": DefaultConverter()}


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

    def __init__(self, converter_name: Optional[str] = None, converter: Optional[PyPDFConverter] = None):
        """
        Create an PyPDFToDocument component.

        :param converter_name:
            The name of a PyPDFConverter instance stored in the CONVERTERS_REGISTRY. Deprecated.
        :param converter:
            An instance of a PyPDFConverter compatible class.
        """
        pypdf_import.check()

        self.converter_name = converter_name
        if converter_name:
            warnings.warn(
                "The `converter_name` parameter is deprecated and will be removed in Haystack 2.3.0. "
                "Please use the `converter` parameter instead.",
                DeprecationWarning,
            )
            try:
                converter = CONVERTERS_REGISTRY[converter_name]
            except KeyError:
                msg = (
                    f"Invalid converter_name: {converter_name}.\n Available converters: {list(CONVERTERS_REGISTRY.keys())}"
                    f"To specify a custom converter, it is recommended to use the `converter` parameter."
                )
                raise ValueError(msg) from KeyError

            self.converter = converter

        elif converter:
            self.converter = converter
        else:
            self.converter = DefaultConverter()

    def to_dict(self):
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(self, converter_name=self.converter_name, converter=self.converter.to_dict())

    @classmethod
    def from_dict(cls, data):
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary with serialized data.

        :returns:
            Deserialized component.
        """
        if data["init_parameters"].get("converter_name"):
            return default_from_dict(cls, data)

        converter_class = deserialize_type(data["init_parameters"]["converter"]["type"])
        data["init_parameters"]["converter"] = converter_class.from_dict(data["init_parameters"]["converter"])
        return default_from_dict(cls, data)

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
                document = self.converter.convert(pdf_reader)
            except Exception as e:
                logger.warning(
                    "Could not read {source} and convert it to Document, skipping. {error}", source=source, error=e
                )
                continue

            merged_metadata = {**bytestream.meta, **metadata}
            document.meta = merged_metadata
            documents.append(document)

        return {"documents": documents}
