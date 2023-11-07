from copy import deepcopy
from typing import List, Literal

from more_itertools import windowed

from haystack.preview import component, Document


@component
class DocumentSplitter:
    """
    Splits a list of text documents into a list of text documents with shorter texts.
    This is useful for splitting documents with long texts that otherwise would not fit into the maximum text length of language models.
    """

    def __init__(
        self, split_by: Literal["word", "sentence", "passage"] = "word", split_length: int = 200, split_overlap: int = 0
    ):
        """
        :param split_by: The unit by which the document should be split. Choose from "word" for splitting by " ",
        "sentence" for splitting by ".", or "passage" for splitting by "\n\n".
        :param split_length: The maximum number of units in each split.
        :param split_overlap: The number of units that each split should overlap.
        """

        self.split_by = split_by
        if split_by not in ["word", "sentence", "passage"]:
            raise ValueError("split_by must be one of 'word', 'sentence' or 'passage'.")
        if split_length <= 0:
            raise ValueError("split_length must be greater than 0.")
        self.split_length = split_length
        if split_overlap < 0:
            raise ValueError("split_overlap must be greater than or equal to 0.")
        self.split_overlap = split_overlap

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        """
        Splits the documents by split_by after split_length units with an overlap of split_overlap units.
        Returns a list of documents with the split texts.
        A metadata field "source_id" is added to each document to keep track of the original document that was split.
        Other metadata are copied from the original document.
        :param documents: The documents to split.
        :return: A list of documents with the split texts.
        """

        if not isinstance(documents, list) or (documents and not isinstance(documents[0], Document)):
            raise TypeError("DocumentSplitter expects a List of Documents as input.")

        split_docs = []
        for doc in documents:
            if doc.content is None:
                raise ValueError(
                    f"DocumentSplitter only works with text documents but document.content for document ID {doc.id} is None."
                )
            units = self._split_into_units(doc.content, self.split_by)
            text_splits = self._concatenate_units(units, self.split_length, self.split_overlap)
            metadata = deepcopy(doc.meta)
            metadata["source_id"] = doc.id
            split_docs += [Document(content=txt, meta=metadata) for txt in text_splits]
        return {"documents": split_docs}

    def _split_into_units(self, text: str, split_by: Literal["word", "sentence", "passage"]) -> List[str]:
        if split_by == "passage":
            split_at = "\n\n"
        elif split_by == "sentence":
            split_at = "."
        elif split_by == "word":
            split_at = " "
        else:
            raise NotImplementedError(
                "DocumentSplitter only supports 'passage', 'sentence' or 'word' split_by options."
            )
        units = text.split(split_at)
        # Add the delimiter back to all units except the last one
        for i in range(len(units) - 1):
            units[i] += split_at
        return units

    def _concatenate_units(self, elements: List[str], split_length: int, split_overlap: int) -> List[str]:
        """
        Concatenates the elements into parts of split_length units.
        """
        text_splits = []
        segments = windowed(elements, n=split_length, step=split_length - split_overlap)
        for seg in segments:
            current_units = [unit for unit in seg if unit is not None]
            txt = "".join(current_units)
            if len(txt) > 0:
                text_splits.append(txt)
        return text_splits
