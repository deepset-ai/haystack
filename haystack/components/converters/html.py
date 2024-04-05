from pathlib import Path
from typing import Any, ClassVar, Dict, List, Literal, Optional, Union

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

    known_extractors: ClassVar[List[str]] = [
        "DefaultExtractor",
        "ArticleExtractor",
        "ArticleSentencesExtractor",
        "LargestContentExtractor",
        "CanolaExtractor",
        "KeepEverythingExtractor",
        "NumWordsRulesExtractor",
    ]

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
        try_others: bool = True,
    ):
        """
        Create an HTMLToDocument component.

        :param
            extractor_type: Name of the extractor class to use. Defaults to `DefaultExtractor`.
            For more information on the different types of extractors,
            see [boilerpy3 documentation](https://github.com/jmriebold/BoilerPy3?tab=readme-ov-file#extractors).
        :param try_others: If `True`, the component will try other extractors if the user chosen extractor fails.
        """
        self.extractor_type = extractor_type
        self.try_others = try_others

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(self, extractor_type=self.extractor_type, try_others=self.try_others)

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

        # Use all extractor types, ensuring user chosen extractor is first, preserve order, avoid duplicates
        extractors_list = (
            list(
                dict.fromkeys(
                    [self.extractor_type, *self.known_extractors]  # User chosen extractor is always tried first
                )
            )
            if self.try_others
            else [self.extractor_type]
        )

        for source, metadata in zip(sources, meta_list):
            try:
                bytestream = get_bytestream_from_source(source=source)
            except Exception as e:
                logger.warning("Could not read {source}. Skipping it. Error: {error}", source=source, error=e)
                continue

            text = None
            for extractor_name in extractors_list:
                extractor_class = getattr(extractors, extractor_name)
                extractor = extractor_class(raise_on_failure=False)
                try:
                    text = extractor.get_content(bytestream.data.decode("utf-8"))
                    if text:
                        break
                except Exception as conversion_e:
                    if self.try_others:
                        logger.warning(
                            "Failed to extract text using {extractor} from {source}. Trying next extractor. Error: {error}",
                            extractor=extractor_name,
                            source=source,
                            error=conversion_e,
                        )
            if not text:
                logger.warning(
                    f"Failed to extract text from {source} using extractors: {extractors_list}. Skipping it.",
                    source=source,
                    extractors_list=extractors_list,
                )
                continue

            document = Document(content=text, meta={**bytestream.meta, **metadata})
            documents.append(document)

        return {"documents": documents}
