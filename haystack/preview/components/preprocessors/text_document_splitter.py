from typing import List, Optional, Dict, Any, Literal, Tuple

from more_itertools import windowed

from haystack.preview import component, Document, default_from_dict, default_to_dict


@component
class TextDocumentSplitter:
    """
    Split a text document into a list of text documents with shorter texts.
    This is useful for splitting documents with long texts that otherwise would not fit into the maximum text length of language models
    """

    def __init__(
        self, split_by: Literal["word", "sentence", "passage"] = "word", split_length: int = 200, split_overlap: int = 0
    ):
        """
        :param split_by: The unit by which the document should be split. Choose from "word", "sentence", "passage".
        :param split_length: The maximum length of each split.
        :param split_overlap: The number of units that each split should overlap.
        """
        self.split_by = split_by
        self.split_length = split_length
        self.split_overlap = split_overlap

    @component.output_types(documents=List[Document])
    def run(
        self,
        document: Document,
        split_by: Optional[Literal["word", "sentence", "passage"]] = None,
        split_length: Optional[int] = None,
        split_overlap: Optional[int] = None,
    ):
        """
        # split the document by split_by after split_length units with an overlap of split_overlap units
        # return a list of documents with the split texts
        :param document: the document to split
        :param split_by: whether to split by word, sentence or passage
        :param split_length: the maximum number of units in each split
        :param split_overlap: the number of units that each split should overlap
        :return: a list of documents with the split texts
        """
        if split_by is None:
            split_by = self.split_by
        if split_length is None:
            split_length = self.split_length
        if split_overlap is None:
            split_overlap = self.split_overlap

        if document.text is None:
            raise ValueError("TextDocumentSplitter only works with text documents but document.text is None.")
        units, split_at = self._split_into_units(document.text, split_by)
        text_splits = self._concatenate_units(units, split_length, split_overlap, split_at)
        documents = [Document(text=txt, metadata=document.metadata) for txt in text_splits]
        return {"documents": documents}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        return default_to_dict(
            self, split_by=self.split_by, split_length=self.split_length, split_overlap=self.split_overlap
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TextDocumentSplitter":
        """
        Deserialize this component from a dictionary.
        """
        return default_from_dict(cls, data)

    def _split_into_units(self, text: str, split_by: Literal["word", "sentence", "passage"]) -> Tuple[List[str], str]:
        if split_by == "passage":
            elements = text.split("\n\n")
            split_at = "\n\n"
        elif split_by == "sentence":
            elements = self._split_sentences(text)
            split_at = "."
        elif split_by == "word":
            elements = text.split(" ")
            split_at = " "
        else:
            raise NotImplementedError("PreProcessor only supports 'passage', 'sentence' or 'word' split_by options.")

        return elements, split_at

    def _concatenate_units(
        self, elements: List[str], split_length: int, split_overlap: int, split_at: str
    ) -> List[str]:
        """
        Concatenates the elements into parts of split_length units.
        """
        text_splits = []
        segments = windowed(elements, n=split_length, step=split_length - split_overlap)
        for seg in segments:
            current_units = [unit for unit in seg if unit is not None]
            txt = split_at.join(current_units)
            if len(txt) > 0:
                text_splits.append(txt)
        return text_splits

    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences. Naive implementation that splits by ".".
        :param text: The text to split.
        :return: The list of sentences.
        """
        return text.split(".")
