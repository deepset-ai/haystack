from typing import List, Optional, Union

from abc import abstractmethod
from haystack.nodes.base import BaseComponent
from haystack.schema import Document


class BasePreProcessor(BaseComponent):
    outgoing_edges = 1

    @abstractmethod
    def process(
        self,
        documents: Union[dict, Document, List[Union[dict, Document]]],
        clean_whitespace: Optional[bool] = True,
        clean_header_footer: Optional[bool] = False,
        clean_empty_lines: Optional[bool] = True,
        remove_substrings: List[str] = [],
        split_by: Optional[str] = "word",
        split_length: Optional[int] = 1000,
        split_overlap: Optional[int] = None,
        split_respect_sentence_boundary: Optional[bool] = True,
    ) -> List[Document]:
        """
        Perform document cleaning and splitting. Takes a single Document or a List of Documents as input and returns a
        list of Documents.
        """
        raise NotImplementedError

    @abstractmethod
    def clean(
        self,
        document: Union[dict, Document],
        clean_whitespace: bool,
        clean_header_footer: bool,
        clean_empty_lines: bool,
        remove_substrings: List[str],
    ) -> Document:
        raise NotImplementedError

    @abstractmethod
    def split(
        self,
        document: Union[dict, Document],
        split_by: str,
        split_length: int,
        split_overlap: int,
        split_respect_sentence_boundary: bool,
    ) -> List[Document]:
        raise NotImplementedError

    def run(  # type: ignore
        self,
        documents: Union[dict, Document, List[Union[dict, Document]]],
        clean_whitespace: Optional[bool] = None,
        clean_header_footer: Optional[bool] = None,
        clean_empty_lines: Optional[bool] = None,
        split_by: Optional[str] = None,
        split_length: Optional[int] = None,
        split_overlap: Optional[int] = None,
        split_respect_sentence_boundary: Optional[bool] = None,
    ):
        processed_documents = self.process(
            documents=documents,
            clean_whitespace=clean_whitespace,
            clean_header_footer=clean_header_footer,
            clean_empty_lines=clean_empty_lines,
            split_by=split_by,
            split_length=split_length,
            split_overlap=split_overlap,
            split_respect_sentence_boundary=split_respect_sentence_boundary,
        )
        result = {"documents": processed_documents}
        return result, "output_1"

    def run_batch(  # type: ignore
        self,
        documents: Union[dict, Document, List[Union[dict, Document]]],
        clean_whitespace: Optional[bool] = None,
        clean_header_footer: Optional[bool] = None,
        clean_empty_lines: Optional[bool] = None,
        split_by: Optional[str] = None,
        split_length: Optional[int] = None,
        split_overlap: Optional[int] = None,
        split_respect_sentence_boundary: Optional[bool] = None,
    ):
        return self.run(
            documents=documents,
            clean_whitespace=clean_whitespace,
            clean_header_footer=clean_header_footer,
            clean_empty_lines=clean_empty_lines,
            split_by=split_by,
            split_length=split_length,
            split_overlap=split_overlap,
            split_respect_sentence_boundary=split_respect_sentence_boundary,
        )
