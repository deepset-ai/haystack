import logging
from pathlib import Path
from typing import List, Union, Dict, Any, Optional
import io

from haystack.lazy_imports import LazyImport
from haystack import component, Document
from haystack.dataclasses import ByteStream
from haystack.components.converters.utils import get_bytestream_from_source


with LazyImport("Run 'pip install tika'") as tika_import:
    from tika import parser as tika_parser

logger = logging.getLogger(__name__)


@component
class TikaDocumentConverter:
    """
    A component for converting files of different types (pdf, docx, html, etc.) to Documents.
    This component uses [Apache Tika](https://tika.apache.org/) for parsing the files and, therefore,
    requires a running Tika server.

    The easiest way to run Tika is to use Docker: `docker run -d -p 127.0.0.1:9998:9998 apache/tika:latest`.
    For more options on running Tika on Docker,
    see the [documentation](https://github.com/apache/tika-docker/blob/main/README.md#usage).
    """

    def __init__(self, tika_url: str = "http://localhost:9998/tika"):
        """
        Create a TikaDocumentConverter component.

        :param tika_url: URL of the Tika server. Default: `"http://localhost:9998/tika"`
        """
        tika_import.check()
        self.tika_url = tika_url

    @component.output_types(documents=List[Document])
    def run(self, sources: List[Union[str, Path, ByteStream]], meta: Optional[List[Dict[str, Any]]] = None):
        """
        Convert files to Documents.

        :param sources: List of file paths or ByteStream objects.
        :param meta: Optional list of metadata to attach to the Documents.
          The length of the list must match the number of sources. Defaults to `None`.
        :return: A dictionary containing a list of Document objects under the 'documents' key.
        """

        documents = []

        if meta is None:
            meta = [{}] * len(sources)
        elif len(sources) != len(meta):
            raise ValueError("The length of the metadata list must match the number of sources.")

        for source, metadata in zip(sources, meta):
            try:
                bytestream = get_bytestream_from_source(source)
            except Exception as e:
                logger.warning("Could not read %s. Skipping it. Error: %s", source, e)
                continue
            try:
                text = tika_parser.from_buffer(io.BytesIO(bytestream.data), serverEndpoint=self.tika_url)["content"]
            except Exception as conversion_e:
                logger.warning("Failed to extract text from %s. Skipping it. Error: %s", source, conversion_e)
                continue

            merged_metadata = {**bytestream.metadata, **metadata}
            document = Document(content=text, meta=merged_metadata)
            documents.append(document)
        return {"documents": documents}
