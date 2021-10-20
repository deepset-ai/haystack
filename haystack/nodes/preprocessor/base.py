from typing import List, Dict, Any, Optional, Union

from haystack.nodes.base import BaseComponent


class BasePreProcessor(BaseComponent):
    outgoing_edges = 1

    def process(
        self,
        documents: Union[dict, List[dict]],
        clean_whitespace: Optional[bool] = True,
        clean_header_footer: Optional[bool] = False,
        clean_empty_lines: Optional[bool] = True,
        split_by: Optional[str] = "word",
        split_length: Optional[int] = 1000,
        split_overlap: Optional[int] = None,
        split_respect_sentence_boundary: Optional[bool] = True,
    ) -> List[dict]:
        """
        Perform document cleaning and splitting. Takes a single document as input and returns a list of documents.
        """
        raise NotImplementedError

    def clean(
        self, document: dict, clean_whitespace: bool, clean_header_footer: bool, clean_empty_lines: bool,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def split(
        self,
        document: dict,
        split_by: str,
        split_length: int,
        split_overlap: int,
        split_respect_sentence_boundary: bool,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def run(  # type: ignore
        self,
        documents: Union[dict, List[dict]],
        clean_whitespace: Optional[bool] = None,
        clean_header_footer: Optional[bool] = None,
        clean_empty_lines: Optional[bool] = None,
        split_by: Optional[str] = None,
        split_length: Optional[int] = None,
        split_overlap: Optional[int] = None,
        split_respect_sentence_boundary: Optional[bool] = None,
    ):
        documents = self.process(
            documents=documents,
            clean_whitespace=clean_whitespace,
            clean_header_footer=clean_header_footer,
            clean_empty_lines=clean_empty_lines,
            split_by=split_by,
            split_length=split_length,
            split_overlap=split_overlap,
            split_respect_sentence_boundary=split_respect_sentence_boundary,
        )
        result = {"documents": documents}
        return result, "output_1"
