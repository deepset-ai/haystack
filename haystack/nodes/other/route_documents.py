from typing import Any, List, Tuple, Dict, Optional, Union
from collections import defaultdict

from haystack.nodes.base import BaseComponent
from haystack.schema import Document


class RouteDocuments(BaseComponent):
    """
    A node to split a list of `Document`s by `content_type` or by the values of a metadata field and route them to
    different nodes.
    """

    # By default (split_by == "content_type"), the node has two outgoing edges.
    outgoing_edges = 2

    def __init__(self, split_by: str = "content_type", metadata_values: Optional[List[str]] = None):
        """
        :param split_by: Field to split the documents by, either `"content_type"` or a metadata field name.
            If this parameter is set to `"content_type"`, the list of `Document`s will be split into a list containing
            only `Document`s of type `"text"` (will be routed to `"output_1"`) and a list containing only `Document`s of
            type `"table"` (will be routed to `"output_2"`).
            If this parameter is set to a metadata field name, you need to specify the parameter `metadata_values` as
            well.
        :param metadata_values: If the parameter `split_by` is set to a metadata field name, you need to provide a list
            of values to group the `Document`s to. `Document`s whose metadata field is equal to the first value of the
            provided list will be routed to `"output_1"`, `Document`s whose metadata field is equal to the second
            value of the provided list will be routed to `"output_2"`, etc.
        """

        if split_by != "content_type" and metadata_values is None:
            raise ValueError(
                "If split_by is set to the name of a metadata field, you must provide metadata_values "
                "to group the documents to."
            )

        super().__init__()

        self.split_by = split_by
        self.metadata_values = metadata_values

    @classmethod
    def _calculate_outgoing_edges(cls, component_params: Dict[str, Any]) -> int:
        split_by = component_params.get("split_by", "content_type")
        metadata_values = component_params.get("metadata_values", None)
        # If we split list of Documents by a metadata field, number of outgoing edges might change
        if split_by != "content_type" and metadata_values is not None:
            return len(metadata_values)
        return 2

    def run(self, documents: List[Document]) -> Tuple[Dict, str]:  # type: ignore
        if self.split_by == "content_type":
            split_documents: Dict[str, List[Document]] = {"output_1": [], "output_2": []}

            for doc in documents:
                if doc.content_type == "text":
                    split_documents["output_1"].append(doc)
                elif doc.content_type == "table":
                    split_documents["output_2"].append(doc)

        else:
            assert isinstance(self.metadata_values, list), (
                "You need to provide metadata_values if you want to split" " a list of Documents by a metadata field."
            )
            split_documents = {f"output_{i+1}": [] for i in range(len(self.metadata_values))}
            for doc in documents:
                current_metadata_value = doc.meta.get(self.split_by, None)
                # Disregard current document if it does not contain the provided metadata field
                if current_metadata_value is not None:
                    try:
                        index = self.metadata_values.index(current_metadata_value)
                    except ValueError:
                        # Disregard current document if current_metadata_value is not in the provided metadata_values
                        continue

                    split_documents[f"output_{index+1}"].append(doc)

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
