from typing import Dict, List, Optional

from haystack import component, Document
from haystack.dataclasses.answer import Answer
from haystack.dataclasses.meta_container import MetaContainer
from haystack.utils.filters import container_matches_filter, convert


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
        component.set_output_types(self, unmatched=List[MetaContainer], **{edge: List[MetaContainer] for edge in rules})

    def run(self, documents: Optional[List[Document]] = None, answers: Optional[List[Answer]] = None):
        """
        Run the MetadataRouter. This method routes the documents to different edges based on their fields content and
        the rules specified during initialization. If a document does not match any of the rules, it is routed to
        a connection named "unmatched".

        :param documents: A list of documents to route to different edges.
        """
        if documents is None and answers is None:
            raise ValueError("Either documents or answers must be provided.")
        elif documents is not None and answers is not None:
            raise ValueError("Only one of documents or answers can be provided.")
        elif documents is not None:
            meta_containers: List[MetaContainer] = documents
        else:
            meta_containers = answers

        unmatched = []
        output: Dict[str, List[MetaContainer]] = {edge: [] for edge in self.rules}

        for meta_container in meta_containers:
            cur_container_matched = False
            for edge, rule in self.rules.items():
                if "operator" not in rule:
                    # Must be a legacy filter, convert it
                    rule = convert(rule)
                if container_matches_filter(rule, meta_container):
                    output[edge].append(meta_container)
                    cur_container_matched = True

            if not cur_container_matched:
                unmatched.append(meta_container)

        output["unmatched"] = unmatched
        return output
