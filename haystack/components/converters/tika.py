import logging
from pathlib import Path
from typing import List, Union
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
    """

    def __init__(self, tika_url: str = "http://localhost:9998/tika"):
        """
        Create a TikaDocumentConverter component.

        :param tika_url: URL of the Tika server. Default: `"http://localhost:9998/tika"`
        """
        tika_import.check()
        self.tika_url = tika_url

    @component.output_types(documents=List[Document])
    def run(self, sources: List[Union[str, Path, ByteStream]]):
        """
        Convert files to Documents.

        :param sources: List of file paths or ByteStream objects.
        """

        documents = []
        for source in sources:
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
            document = Document(content=text)
            documents.append(document)
        return {"documents": documents}

