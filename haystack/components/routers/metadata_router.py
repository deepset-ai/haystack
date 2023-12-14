from typing import Any, Dict, List, Optional, Sequence

from haystack import component, Document
from haystack.dataclasses.answer import Answer
from haystack.dataclasses.meta_dataclass import MetaDataclass
from haystack.utils.filters import data_object_matches_filter, convert


@component
class MetadataRouter:
    """
    A component that routes documents to different connections based on the content of their fields.
    """

    def __init__(self, rules: Dict[str, Dict]):
        """
        Initialize the MetadataRouter.

        :param rules: A dictionary of rules that specify which edge to route a document to based on its metadata.
                      The keys of the dictionary are the names of the output connections, and the values are dictionaries that
                      follow the format of filtering expressions in Haystack. For example:
                      ```python
                      {
                        "edge_1": {
                            "operator": "AND",
                            "conditions": [
                                {"field": "meta.created_at", "operator": ">=", "value": "2023-01-01"},
                                {"field": "meta.created_at", "operator": "<", "value": "2023-04-01"},
                            ],
                        },
                        "edge_2": {
                            "operator": "AND",
                            "conditions": [
                                {"field": "meta.created_at", "operator": ">=", "value": "2023-04-01"},
                                {"field": "meta.created_at", "operator": "<", "value": "2023-07-01"},
                            ],
                        },
                        "edge_3": {
                            "operator": "AND",
                            "conditions": [
                                {"field": "meta.created_at", "operator": ">=", "value": "2023-07-01"},
                                {"field": "meta.created_at", "operator": "<", "value": "2023-10-01"},
                            ],
                        },
                        "edge_4": {
                            "operator": "AND",
                            "conditions": [
                                {"field": "meta.created_at", "operator": ">=", "value": "2023-10-01"},
                                {"field": "meta.created_at", "operator": "<", "value": "2024-01-01"},
                            ],
                        },
                    }
                    ```
        """
        self.rules = rules
        component.set_output_types(self, unmatched=List[MetaDataclass], **{edge: List[MetaDataclass] for edge in rules})

    def run(self, documents: Optional[List[Document]] = None, answers: Optional[List[Answer]] = None):
        """
        Run the MetadataRouter. This method routes the documents to different edges based on their fields content and
        the rules specified during initialization. If a document does not match any of the rules, it is routed to
        a connection named "unmatched".

        :param documents: A list of documents to route to different edges.
        """
        if documents is not None and answers is not None:
            raise ValueError("Only one of documents or answers can be provided.")

        data_objects: Sequence[MetaDataclass]
        if documents is not None:
            data_objects = documents
        elif answers is not None:
            data_objects = answers
        else:
            raise ValueError("Either documents or answers must be provided.")

        unmatched = []
        output: Dict[str, List[Any]] = {edge: [] for edge in self.rules}

        for data_object in data_objects:
            cur_data_object_matched = False
            for edge, rule in self.rules.items():
                if "operator" not in rule:
                    # Must be a legacy filter, convert it
                    rule = convert(rule)
                if data_object_matches_filter(rule, data_object):
                    output[edge].append(data_object)
                    cur_data_object_matched = True

            if not cur_data_object_matched:
                unmatched.append(data_object)

        output["unmatched"] = unmatched
        return output
