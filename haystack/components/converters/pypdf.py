# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import io
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


@component
class PyPDFToDocument:
    """
    Converts PDF files to documents your pipeline can query.

    This component uses converters compatible with the PyPDF library.
    If no converter is provided, uses a default text extraction converter.
    You can attach metadata to the resulting documents.

    ### Usage example

    ```python
    from haystack.components.converters.pypdf import PyPDFToDocument

    converter = PyPDFToDocument()
    results = converter.run(sources=["sample.pdf"], meta={"date_added": datetime.now().isoformat()})
    documents = results["documents"]
    print(documents[0].content)
    # 'This is a text from the PDF file.'
    ```
    """

    def __init__(self, converter: Optional[PyPDFConverter] = None):
        """
        Create an PyPDFToDocument component.

        :param converter:
            An instance of a PyPDFConverter compatible class.
        """
        pypdf_import.check()

        self.converter = converter or DefaultConverter()

    def to_dict(self):
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(self, converter=self.converter.to_dict())

    @classmethod
    def from_dict(cls, data):
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary with serialized data.

        :returns:
            Deserialized component.
        """
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
        Converts PDF files to documents.

        :param sources:
            List of file paths or ByteStream objects to convert.
        :param meta:
            Optional metadata to attach to the documents.
            This value can be a list of dictionaries or a single dictionary.
            If it's a single dictionary, its content is added to the metadata of all produced documents.
            If it's a list, its length must match the number of sources, as they are zipped together.
            For ByteStream objects, their `meta` is added to the output documents.

        :returns:
            A dictionary with the following keys:
            - `documents`: A list of converted documents.
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
