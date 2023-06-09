from typing import Any, List, Tuple, Dict, Optional, Union
from collections import defaultdict
import logging

from haystack.nodes.base import BaseComponent
from haystack.schema import Document

logger = logging.getLogger(__name__)


class RouteDocuments(BaseComponent):
    """
    A node to split a list of `Document`s by `content_type` or by the values of a metadata field and route them to
    different nodes.
    """

    # By default (split_by == "content_type"), the node has two outgoing edges.
    outgoing_edges = 2

    def __init__(
        self,
        split_by: str = "content_type",
        metadata_values: Optional[Union[List[str], List[List[str]]]] = None,
        return_remaining: bool = False,
    ):
        """
        :param split_by: Field to split the documents by, either `"content_type"` or a metadata field name.
            If this parameter is set to `"content_type"`, the list of `Document`s will be split into a list containing
            only `Document`s of type `"text"` (will be routed to `"output_1"`) and a list containing only `Document`s of
            type `"table"` (will be routed to `"output_2"`).
            If this parameter is set to a metadata field name, you need to specify the parameter `metadata_values` as
            well.
         :param metadata_values: A list of values to group `Document`s by metadata field. If the parameter `split_by`
            is set to a metadata field name, you must provide a list of values (or a list of lists of values) to
            group the `Document`s by.
            If `metadata_values` is a list of strings, then the `Document`s whose metadata field is equal to the
            corresponding value will be routed to the output with the same index.
            If `metadata_values` is a list of lists, then the `Document`s whose metadata field is equal to the first
            value of the provided sublist will be routed to `"output_1"`, the `Document`s whose metadata field is equal
            to the second value of the provided sublist will be routed to `"output_2"`, and so on.
        :param return_remaining: Whether to return all remaining documents that don't match the `split_by` or
            `metadata_values` into an additional output route. This additional output route will be indexed to plus one
             of the previous last output route. For example, if there would normally be `"output_1"` and `"output_2"`
             when return_remaining  is False, then when return_remaining is True the additional output route would be
             `"output_3"`.
        """
        super().__init__()

        self.split_by = split_by
        self.metadata_values = metadata_values
        self.return_remaining = return_remaining

        if self.split_by != "content_type":
            if self.metadata_values is None or len(self.metadata_values) == 0:
                raise ValueError(
                    "If split_by is set to the name of a metadata field, provide metadata_values if you want to split "
                    "a list of Documents by a metadata field."
                )

    @classmethod
    def _calculate_outgoing_edges(cls, component_params: Dict[str, Any]) -> int:
        split_by = component_params.get("split_by", "content_type")
        metadata_values = component_params.get("metadata_values", None)
        return_remaining = component_params.get("return_remaining", False)

        # If we split list of Documents by a metadata field, number of outgoing edges might change
        if split_by != "content_type" and metadata_values is not None:
            num_edges = len(metadata_values)
        else:
            num_edges = 2

        if return_remaining:
            num_edges += 1
        return num_edges

    def _split_by_content_type(self, documents: List[Document]) -> Dict[str, List[Document]]:
        mapping = {"text": "output_1", "table": "output_2"}
        split_documents: Dict[str, List[Document]] = {"output_1": [], "output_2": [], "output_3": []}
        for doc in documents:
            output_route = mapping.get(doc.content_type, "output_3")
            split_documents[output_route].append(doc)

        if not self.return_remaining:
            # Used to avoid unnecessarily calculating other_content_types depending on logging level
            if logger.isEnabledFor(logging.WARNING) and len(split_documents["output_3"]) > 0:
                other_content_types = {x.content_type for x in split_documents["output_3"]}
                logger.warning(
                    "%s document(s) were skipped because they have content type(s) %s. Only the content "
                    "types 'text' and 'table' are routed.",
                    len(split_documents["output_3"]),
                    other_content_types,
                )
            del split_documents["output_3"]

        return split_documents

    def _get_metadata_values_index(self, metadata_values: Union[List[str], List[List[str]]], value: str) -> int:
        for idx, item in enumerate(metadata_values):
            if isinstance(item, list):
                if value in item:
                    return idx
            else:
                if value == item:
                    return idx
        return len(metadata_values)

    def _split_by_metadata_values(
        self, metadata_values: Union[List, List[List]], documents: List[Document]
    ) -> Dict[str, List[Document]]:
        # We need also to keep track of the excluded documents so we add 2 to the number of metadata_values
        output_keys = [f"output_{i}" for i in range(1, len(metadata_values) + 2)]
        split_documents: Dict[str, List[Document]] = {k: [] for k in output_keys}
        # This is the key used for excluded documents
        remaining_key = output_keys[-1]

        for doc in documents:
            current_metadata_value = doc.meta.get(self.split_by, remaining_key)
            index = self._get_metadata_values_index(metadata_values, current_metadata_value)
            output = output_keys[index]
            split_documents[output].append(doc)

        if not self.return_remaining:
            if len(split_documents[remaining_key]) > 0:
                logger.warning(
                    "%s documents were skipped because they were either missing the metadata field '%s' or the"
                    " corresponding metadata value is not included in `metadata_values`.",
                    len(split_documents[remaining_key]),
                    self.split_by,
                )
            del split_documents[remaining_key]

        return split_documents

    def run(self, documents: List[Document]) -> Tuple[Dict, str]:  # type: ignore
        if self.split_by == "content_type":
            split_documents = self._split_by_content_type(documents)
        elif self.metadata_values:
            split_documents = self._split_by_metadata_values(self.metadata_values, documents)
        else:
            raise ValueError(
                "If split_by is set to the name of a metadata field, provide metadata_values if you want to split "
                "a list of Documents by a metadata field."
            )
        return split_documents, "split"

    def run_batch(self, documents: Union[List[Document], List[List[Document]]]) -> Tuple[Dict, str]:  # type: ignore
        if isinstance(documents[0], Document):
            return self.run(documents)  # type: ignore
        else:
            split_documents = defaultdict(list)
            for doc_list in documents:
                results, _ = self.run(documents=doc_list)  # type: ignore
                for key in results:
                    split_documents[key].append(results[key])
            return split_documents, "split"
