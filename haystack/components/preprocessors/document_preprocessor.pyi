from typing import Any, Dict, List

from haystack import super_component
from haystack.dataclasses import Document

@super_component
class DocumentPreprocessor:
    """
    A SuperComponent that first splits and then cleans documents.

    This component consists of a DocumentSplitter followed by a DocumentCleaner in a single pipeline.
    It takes a list of documents as input and returns a processed list of documents.
    """
    def __init__(
        self,
        *,
        split_by: str = "word",
        split_length: int = 250,
        split_overlap: int = 0,
        respect_sentence_boundary: bool = False,
        language: str = "en",
        remove_empty_lines: bool = True,
        remove_extra_whitespaces: bool = True,
        remove_repeated_substrings: bool = False,
    ) -> None: ...
    def to_dict(self) -> Dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentPreprocessor": ...
    def run(self, documents: List[Document]) -> Dict[str, List[Document]]: ...
