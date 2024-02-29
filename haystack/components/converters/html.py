from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from boilerpy3 import extractors

from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.components.converters.utils import get_bytestream_from_source, normalize_metadata
from haystack.dataclasses import ByteStream

logger = logging.getLogger(__name__)


@component
class HTMLToDocument:
    """
    Converts an HTML file to a Document.

    Usage example:
    ```python
    from haystack.components.converters import HTMLToDocument

    converter = HTMLToDocument()
    results = converter.run(sources=["path/to/sample.html"])
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

        :param
            extractor_type: Name of the extractor class to use. Defaults to `DefaultExtractor`.
            For more information on the different types of extractors,
            see [boilerpy3 documentation](https://github.com/jmriebold/BoilerPy3?tab=readme-ov-file#extractors).
        """
        self.extractor_type = extractor_type

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(self, extractor_type=self.extractor_type)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HTMLToDocument":
        """
        Deserializes the component from a dictionary.

        :param data:
            The dictionary to deserialize from.
        :returns:
            The deserialized component.
        """
        return default_from_dict(cls, data)

    @component.output_types(documents=List[Document])
    def run(
        self,
        sources: List[Union[str, Path, ByteStream]],
        meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    ):
        """
        Converts a list of HTML files to Documents.

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

        extractor_class = getattr(extractors, self.extractor_type)
        extractor = extractor_class(raise_on_failure=False)

        for source, metadata in zip(sources, meta_list):
            try:
                bytestream = get_bytestream_from_source(source=source)
            except Exception as e:
                logger.warning("Could not read {source}. Skipping it. Error: {error}", source=source, error=e)
                continue
            try:
                file_content = bytestream.data.decode("utf-8")
                text = extractor.get_content(file_content)
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
