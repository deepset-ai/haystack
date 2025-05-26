# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional

from haystack import Document, Pipeline, default_from_dict, default_to_dict, super_component
from haystack.components.preprocessors.document_cleaner import DocumentCleaner
from haystack.components.preprocessors.document_splitter import DocumentSplitter, Language
from haystack.utils import deserialize_callable, serialize_callable


@super_component
class DocumentPreprocessor:
    """
    A SuperComponent that first splits and then cleans documents.

    This component consists of a DocumentSplitter followed by a DocumentCleaner in a single pipeline.
    It takes a list of documents as input and returns a processed list of documents.

    Usage example:
    ```python
    from haystack import Document
    from haystack.components.preprocessors import DocumentPreprocessor

    doc = Document(content="I love pizza!")
    preprocessor = DocumentPreprocessor()
    result = preprocessor.run(documents=[doc])
    print(result["documents"])
    ```
    """

    def __init__(  # noqa: PLR0913 (too-many-arguments)
        self,
        *,
        # --- DocumentSplitter arguments ---
        split_by: Literal["function", "page", "passage", "period", "word", "line", "sentence"] = "word",
        split_length: int = 250,
        split_overlap: int = 0,
        split_threshold: int = 0,
        splitting_function: Optional[Callable[[str], List[str]]] = None,
        respect_sentence_boundary: bool = False,
        language: Language = "en",
        use_split_rules: bool = True,
        extend_abbreviations: bool = True,
        # --- DocumentCleaner arguments ---
        remove_empty_lines: bool = True,
        remove_extra_whitespaces: bool = True,
        remove_repeated_substrings: bool = False,
        keep_id: bool = False,
        remove_substrings: Optional[List[str]] = None,
        remove_regex: Optional[str] = None,
        unicode_normalization: Optional[Literal["NFC", "NFKC", "NFD", "NFKD"]] = None,
        ascii_only: bool = False,
    ) -> None:
        """
        Initialize a DocumentPreProcessor that first splits and then cleans documents.

        **Splitter Parameters**:
        :param split_by: The unit of splitting: "function", "page", "passage", "period", "word", "line", or "sentence".
        :param split_length: The maximum number of units (words, lines, pages, and so on) in each split.
        :param split_overlap: The number of overlapping units between consecutive splits.
        :param split_threshold: The minimum number of units per split. If a split is smaller than this, it's merged
            with the previous split.
        :param splitting_function: A custom function for splitting if `split_by="function"`.
        :param respect_sentence_boundary: If `True`, splits by words but tries not to break inside a sentence.
        :param language: Language used by the sentence tokenizer if `split_by="sentence"` or
            `respect_sentence_boundary=True`.
        :param use_split_rules: Whether to apply additional splitting heuristics for the sentence splitter.
        :param extend_abbreviations: Whether to extend the sentence splitter with curated abbreviations for certain
            languages.

        **Cleaner Parameters**:
        :param remove_empty_lines: If `True`, removes empty lines.
        :param remove_extra_whitespaces: If `True`, removes extra whitespaces.
        :param remove_repeated_substrings: If `True`, removes repeated substrings like headers/footers across pages.
        :param keep_id: If `True`, keeps the original document IDs.
        :param remove_substrings: A list of strings to remove from the document content.
        :param remove_regex: A regex pattern whose matches will be removed from the document content.
        :param unicode_normalization: Unicode normalization form to apply to the text, for example `"NFC"`.
        :param ascii_only: If `True`, converts text to ASCII only.
        """
        # Store arguments for serialization
        self.remove_empty_lines = remove_empty_lines
        self.remove_extra_whitespaces = remove_extra_whitespaces
        self.remove_repeated_substrings = remove_repeated_substrings
        self.keep_id = keep_id
        self.remove_substrings = remove_substrings
        self.remove_regex = remove_regex
        self.unicode_normalization = unicode_normalization
        self.ascii_only = ascii_only

        self.split_by = split_by
        self.split_length = split_length
        self.split_overlap = split_overlap
        self.split_threshold = split_threshold
        self.splitting_function = splitting_function
        self.respect_sentence_boundary = respect_sentence_boundary
        self.language = language
        self.use_split_rules = use_split_rules
        self.extend_abbreviations = extend_abbreviations

        # Instantiate sub-components
        splitter = DocumentSplitter(
            split_by=self.split_by,
            split_length=self.split_length,
            split_overlap=self.split_overlap,
            split_threshold=self.split_threshold,
            splitting_function=self.splitting_function,
            respect_sentence_boundary=self.respect_sentence_boundary,
            language=self.language,
            use_split_rules=self.use_split_rules,
            extend_abbreviations=self.extend_abbreviations,
        )

        cleaner = DocumentCleaner(
            remove_empty_lines=self.remove_empty_lines,
            remove_extra_whitespaces=self.remove_extra_whitespaces,
            remove_repeated_substrings=self.remove_repeated_substrings,
            keep_id=self.keep_id,
            remove_substrings=self.remove_substrings,
            remove_regex=self.remove_regex,
            unicode_normalization=self.unicode_normalization,
            ascii_only=self.ascii_only,
        )

        # Build the Pipeline
        pp = Pipeline()

        pp.add_component("splitter", splitter)
        pp.add_component("cleaner", cleaner)

        # Connect the splitter output to cleaner
        pp.connect("splitter.documents", "cleaner.documents")
        self.pipeline = pp

        # Define how pipeline inputs/outputs map to sub-component inputs/outputs
        self.input_mapping = {
            # The pipeline input "documents" feeds into "splitter.documents"
            "documents": ["splitter.documents"]
        }
        # The pipeline output "documents" comes from "cleaner.documents"
        self.output_mapping = {"cleaner.documents": "documents"}

    if TYPE_CHECKING:
        # fake method, never executed, but static analyzers will not complain about missing method
        def run(self, *, documents: List[Document]) -> dict[str, list[Document]]:  # noqa: D102
            ...
        def warm_up(self) -> None:  # noqa: D102
            ...

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize SuperComponent to a dictionary.

        :return:
            Dictionary with serialized data.
        """
        splitting_function = None
        if self.splitting_function is not None:
            splitting_function = serialize_callable(self.splitting_function)

        return default_to_dict(
            self,
            remove_empty_lines=self.remove_empty_lines,
            remove_extra_whitespaces=self.remove_extra_whitespaces,
            remove_repeated_substrings=self.remove_repeated_substrings,
            keep_id=self.keep_id,
            remove_substrings=self.remove_substrings,
            remove_regex=self.remove_regex,
            unicode_normalization=self.unicode_normalization,
            ascii_only=self.ascii_only,
            split_by=self.split_by,
            split_length=self.split_length,
            split_overlap=self.split_overlap,
            split_threshold=self.split_threshold,
            splitting_function=splitting_function,
            respect_sentence_boundary=self.respect_sentence_boundary,
            language=self.language,
            use_split_rules=self.use_split_rules,
            extend_abbreviations=self.extend_abbreviations,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentPreprocessor":
        """
        Deserializes the SuperComponent from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized SuperComponent.
        """
        splitting_function = data["init_parameters"].get("splitting_function", None)
        if splitting_function:
            data["init_parameters"]["splitting_function"] = deserialize_callable(splitting_function)
        return default_from_dict(cls, data)
