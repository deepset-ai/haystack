import logging
from typing import List, Union
from pathlib import Path

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
    """

    def __init__(self):
        """
        Initializes the HTMLToDocument component.
        """
        boilerpy3_import.check()

    @component.output_types(documents=List[Document])
    def run(self, sources: List[Union[str, Path, ByteStream]]):
        """
        Converts a list of HTML files to Documents.

        :param sources: Paths to HTML files.
        :return: List of converted Documents.
        """
        documents = []
        extractor = extractors.ArticleExtractor(raise_on_failure=False)
        for source in sources:
            try:
                file_content = self._extract_content(source)
            except Exception as e:
                logger.warning("Could not read %s. Skipping it. Error: %s", source, e)
                continue
            try:
                text = extractor.get_content(file_content)
            except Exception as conversion_e:  # Consider specifying the expected exception type(s) here
                logger.warning("Failed to extract text from %s. Skipping it. Error: %s", source, conversion_e)
                continue

            document = Document(text=text)
            documents.append(document)

        return {"documents": documents}

    def _extract_content(self, source: Union[str, Path, ByteStream]) -> str:
        """
        Extracts content from the given data source
        :param source: The data source to extract content from.
        :return: The extracted content.
        """
        if isinstance(source, (str, Path)):
            with open(source) as text_file:
                return text_file.read()
        if isinstance(source, ByteStream):
            return source.data.decode("utf-8")

        raise ValueError(f"Unsupported source type: {type(source)}")
