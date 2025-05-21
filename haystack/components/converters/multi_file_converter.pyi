from typing import Any, Dict, List, Optional

from haystack import super_component
from haystack.dataclasses import Document

@super_component
class MultiFileConverter:
    """
    A SuperComponent that converts multiple files into documents.

    This component can handle various file formats and convert them into Haystack Document objects.
    It supports multiple input files and returns a list of processed documents.
    """
    def __init__(
        self,
        *,
        supported_formats: Optional[List[str]] = None,
        encoding: str = "utf-8",
        remove_empty_lines: bool = True,
        remove_extra_whitespace: bool = True,
    ) -> None: ...
    def to_dict(self) -> Dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultiFileConverter": ...
    def run(self, files: List[str]) -> Dict[str, List[Document]]: ...
