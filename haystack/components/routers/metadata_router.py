from typing import Dict, List

from haystack import component, Document
from haystack.utils.filters import document_matches_filter, convert


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
        component.set_output_types(self, unmatched=List[Document], **{edge: List[Document] for edge in rules})

    def run(self, documents: List[Document]):
        """
        Run the MetadataRouter. This method routes the documents to different edges based on their fields content and
        the rules specified during initialization. If a document does not match any of the rules, it is routed to
        a connection named "unmatched".

        :param documents: A list of documents to route to different edges.
        """
        unmatched_documents = []
        output: Dict[str, List[Document]] = {edge: [] for edge in self.rules}

        for document in documents:
            cur_document_matched = False
            for edge, rule in self.rules.items():
                if "operator" not in rule:
                    # Must be a legacy filter, convert it
                    rule = convert(rule)
                if document_matches_filter(rule, document):
                    output[edge].append(document)
                    cur_document_matched = True

            if not cur_document_matched:
                unmatched_documents.append(document)

        output["unmatched"] = unmatched_documents
        return output
