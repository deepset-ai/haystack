import logging
from typing import List, Optional, Dict, Any, Union
from pathlib import Path

from haystack.preview.lazy_imports import LazyImport
from haystack.preview import Document, component, default_to_dict, default_from_dict

with LazyImport("Run 'pip install boilerpy3'") as boilerpy3_import:
    from boilerpy3 import extractors


logger = logging.getLogger(__name__)


@component
class HTMLToDocument:
    """
    A component for converting an HTML file to a Document.
    """

    def __init__(self, id_hash_keys: Optional[List[str]] = None):
        """
        Create a HTMLToDocument component.

        :param id_hash_keys: Generate the Document ID from a custom list of strings that refer to the Document's
            attributes. Default: `None`
        """
        boilerpy3_import.check()
        self.id_hash_keys = id_hash_keys or []

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        return default_to_dict(self, id_hash_keys=self.id_hash_keys)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HTMLToDocument":
        """
        Deserialize this component from a dictionary.
        """
        return default_from_dict(cls, data)

    @component.output_types(documents=List[Document])
    def run(self, paths: List[Union[str, Path]]):
        """
        Convert HTML files to Documents.

        :param paths: A list of paths to HTML files.
        :return: A list of Documents.
        """
        documents = []
        extractor = extractors.ArticleExtractor(raise_on_failure=False)
        for path in paths:
            try:
                file_content = extractor.read_from_file(path)
            except Exception as e:
                logger.warning("Could not read file %s. Skipping it. Error message: %s", path, e)
                continue
            # although raise_on_failure is set to False, the extractor can still raise an exception
            try:
                text = extractor.get_content(file_content)
            except Exception as conversion_e:
                logger.warning("Could not extract raw txt from %s. Skipping it. Error message: %s", path, conversion_e)
                continue

            document = Document(text=text, id_hash_keys=self.id_hash_keys)
            documents.append(document)

        return {"documents": documents}
