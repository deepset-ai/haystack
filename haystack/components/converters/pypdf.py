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
from haystack.utils.base_serialization import deserialize_class_instance, serialize_class_instance

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

        self.converter = converter

    def to_dict(self):
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self, converter=(serialize_class_instance(self.converter) if self.converter is not None else None)
        )

    @classmethod
    def from_dict(cls, data):
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary with serialized data.

        :returns:
            Deserialized component.
        """
        custom_converter_data = data["init_parameters"]["converter"]
        if custom_converter_data is not None:
            data["init_parameters"]["converter"] = deserialize_class_instance(custom_converter_data)

        return default_from_dict(cls, data)

    def _default_convert(self, reader: "PdfReader") -> Document:
        text = "\f".join(page.extract_text() for page in reader.pages)
        return Document(content=text)

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
                document = (
                    self._default_convert(pdf_reader) if self.converter is None else self.converter.convert(pdf_reader)
                )
            except Exception as e:
                logger.warning(
                    "Could not read {source} and convert it to Document, skipping. {error}", source=source, error=e
                )
                continue

            merged_metadata = {**bytestream.meta, **metadata}
            document.meta = merged_metadata
            documents.append(document)

        return {"documents": documents}
