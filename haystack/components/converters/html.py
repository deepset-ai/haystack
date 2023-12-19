import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Literal
from boilerpy3 import extractors

from haystack import Document, component
from haystack.dataclasses import ByteStream
from haystack.components.converters.utils import get_bytestream_from_source

logger = logging.getLogger(__name__)


@component
class HTMLToDocument:
    """
    Converts an HTML file to a Document.

    Usage example:
    ```python
    from haystack.components.converters.html import HTMLToDocument

    converter = HTMLToDocument()
    results = converter.run(sources=["sample.html"])
    documents = results["documents"]
    print(documents[0].content)
    # 'This is a text from the HTML file.'
    ```

    """

    def __init__(
        self,
        extractor_type: Literal[
            "DefaultExtractor",
            "ArticleExtractor",
            "ArticleSentencesExtractor",
            "LargestContentExtractor",
            "CanolaExtractor",
            "KeepEverythingExtractor",
            "NumWordsRulesExtractor",
        ] = "DefaultExtractor",
    ):
        """
        Create an HTMLToDocument component.

        :param extractor_type: The type of boilerpy3 extractor to use. Defaults to `DefaultExtractor`.
          For more information on the different types of extractors,
          see [boilerpy3 documentation](https://github.com/jmriebold/BoilerPy3?tab=readme-ov-file#extractors).
        """
        self.extractor_type = extractor_type

    @component.output_types(documents=List[Document])
    def run(self, sources: List[Union[str, Path, ByteStream]], meta: Optional[List[Dict[str, Any]]] = None):
        """
        Converts a list of HTML files to Documents.

        :param sources: List of HTML file paths or ByteStream objects.
        :param meta: Optional list of metadata to attach to the Documents.
          The length of the list must match the number of sources. Defaults to `None`.
        :return: A dictionary containing a list of Document objects under the 'documents' key.
        """

        documents = []

        if meta is None:
            meta = [{}] * len(sources)
        elif len(sources) != len(meta):
            raise ValueError("The length of the metadata list must match the number of sources.")

        extractor_class = getattr(extractors, self.extractor_type)
        extractor = extractor_class(raise_on_failure=False)

        for source, metadata in zip(sources, meta):
            try:
                bytestream = get_bytestream_from_source(source=source)
            except Exception as e:
                logger.warning("Could not read %s. Skipping it. Error: %s", source, e)
                continue
            try:
                file_content = bytestream.data.decode("utf-8")
                text = extractor.get_content(file_content)
            except Exception as conversion_e:
                logger.warning("Failed to extract text from %s. Skipping it. Error: %s", source, conversion_e)
                continue

            merged_metadata = {**bytestream.metadata, **metadata}
            document = Document(content=text, meta=merged_metadata)
            documents.append(document)

        return {"documents": documents}
