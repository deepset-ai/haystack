import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from haystack import Document, component, logging
from haystack.components.converters.utils import get_bytestream_from_source, normalize_metadata
from haystack.dataclasses import ByteStream
from haystack.lazy_imports import LazyImport

with LazyImport("Run 'pip install tika'") as tika_import:
    from tika import parser as tika_parser

logger = logging.getLogger(__name__)


@component
class TikaDocumentConverter:
    """
    Converts files of different types to Documents.

    This component uses [Apache Tika](https://tika.apache.org/) for parsing the files and, therefore,
    requires a running Tika server.
    For more options on running Tika,
    see the [official documentation](https://github.com/apache/tika-docker/blob/main/README.md#usage).

    Usage example:
    ```python
    from haystack.components.converters.tika import TikaDocumentConverter

    converter = TikaDocumentConverter()
    results = converter.run(
        sources=["sample.docx", "my_document.rtf", "archive.zip"],
        meta={"date_added": datetime.now().isoformat()}
    )
    documents = results["documents"]
    print(documents[0].content)
    # 'This is a text from the docx file.'
    ```
    """

    def __init__(self, tika_url: str = "http://localhost:9998/tika"):
        """
        Create a TikaDocumentConverter component.

        :param tika_url:
            Tika server URL.
        """
        tika_import.check()
        self.tika_url = tika_url

    @component.output_types(documents=List[Document])
    def run(
        self,
        sources: List[Union[str, Path, ByteStream]],
        meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    ):
        """
        Converts files to Documents.

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
        meta_list = normalize_metadata(meta=meta, sources_count=len(sources))

        for source, metadata in zip(sources, meta_list):
            try:
                bytestream = get_bytestream_from_source(source)
            except Exception as e:
                logger.warning("Could not read {source}. Skipping it. Error: {error}", source=source, error=e)
                continue
            try:
                text = tika_parser.from_buffer(io.BytesIO(bytestream.data), serverEndpoint=self.tika_url)["content"]
            except Exception as conversion_e:
                logger.warning(
                    "Failed to extract text from {source}. Skipping it. Error: {error}",
                    source=source,
                    error=conversion_e,
                )
                continue

            merged_metadata = {**bytestream.meta, **metadata}
            document = Document(content=text, meta=merged_metadata)
            documents.append(document)
        return {"documents": documents}
