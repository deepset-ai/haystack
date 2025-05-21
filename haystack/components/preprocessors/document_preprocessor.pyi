# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Dict, List, Literal, Optional

from haystack import super_component
from haystack.components.preprocessors.sentence_tokenizer import Language
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
        split_by: Literal["function", "page", "passage", "period", "word", "line", "sentence"] = "word",
        split_length: int = 250,
        split_overlap: int = 0,
        split_threshold: int = 0,
        splitting_function: Optional[Callable[[str], List[str]]] = None,
        respect_sentence_boundary: bool = False,
        language: Language = "en",
        use_split_rules: bool = True,
        extend_abbreviations: bool = True,
        remove_empty_lines: bool = True,
        remove_extra_whitespaces: bool = True,
        remove_repeated_substrings: bool = False,
        keep_id: bool = False,
        remove_substrings: Optional[List[str]] = None,
        remove_regex: Optional[str] = None,
        unicode_normalization: Optional[Literal["NFC", "NFKC", "NFD", "NFKD"]] = None,
        ascii_only: bool = False,
    ) -> None: ...
    def to_dict(self) -> Dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentPreprocessor": ...
    def run(self, documents: List[Document]) -> Dict[str, List[Document]]: ...
