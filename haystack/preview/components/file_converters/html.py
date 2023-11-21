import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from haystack.preview import Document, component
from haystack.preview.dataclasses import ByteStream
from haystack.preview.lazy_imports import LazyImport

logger = logging.getLogger(__name__)

with LazyImport("Run 'pip install boilerpy3'") as boilerpy3_import:
    from boilerpy3 import extractors


@component
class HTMLToDocument:
    """
    Converts an HTML file to a Document.

    Usage example:
    ```python
    from haystack.preview.components.file_converters.html import HTMLToDocument

    converter = HTMLToDocument()
    results = converter.run(sources=["sample.html"])
    documents = results["documents"]
    print(documents[0].content)
    # 'This is a text from the HTML file.'
    ```

    """

    def __init__(self):
        """
        Initializes the HTMLToDocument component.
        """
        boilerpy3_import.check()

    @component.output_types(documents=List[Document])
    def run(self, sources: List[Union[str, Path, ByteStream]], meta: Optional[List[Dict[str, Any]]] = None):
        """
        Converts a list of HTML files to Documents.

        :param sources: List of HTML file paths or ByteStream objects.
        :param meta: Optional list of metadata to attach to the Documents.
        The length of the list must match the number of sources. Defaults to `None`.
        :return: List of converted Documents.
        """

        documents = []

        # Create metadata placeholders if not provided
        if meta:
            if len(sources) != len(meta):
                raise ValueError("The length of the metadata list must match the number of sources.")
        else:
            meta = [{}] * len(sources)

        extractor = extractors.ArticleExtractor(raise_on_failure=False)

        for source, metadata in zip(sources, meta):
            try:
                file_content, extracted_meta = self._extract_content(source)
            except Exception as e:
                logger.warning("Could not read %s. Skipping it. Error: %s", source, e)
                continue
            try:
                text = extractor.get_content(file_content)
            except Exception as conversion_e:  # Consider specifying the expected exception type(s) here
                logger.warning("Failed to extract text from %s. Skipping it. Error: %s", source, conversion_e)
                continue

            # Merge metadata received from ByteStream with supplied metadata
            if extracted_meta:
                # Supplied metadata overwrites metadata from ByteStream for overlapping keys.
                metadata = {**extracted_meta, **metadata}
            document = Document(content=text, meta=metadata)
            documents.append(document)

        return {"documents": documents}

    def _extract_content(self, source: Union[str, Path, ByteStream]) -> tuple:
        """
        Extracts content from the given data source
        :param source: The data source to extract content from.
        :return: The extracted content and metadata.
        """
        if isinstance(source, (str, Path)):
            with open(source) as text_file:
                return (text_file.read(), None)
        if isinstance(source, ByteStream):
            return (source.data.decode("utf-8"), source.metadata)

        raise ValueError(f"Unsupported source type: {type(source)}")
