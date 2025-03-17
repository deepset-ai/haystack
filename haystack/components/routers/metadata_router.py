# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List

from haystack import Document, component
from haystack.utils.filters import document_matches_filter


@component
class MetadataRouter:
    """
    Routes documents to different connections based on their metadata fields.

    Specify the routing rules in the `init` method.
    If a document does not match any of the rules, it's routed to a connection named "unmatched".

    ### Usage example

    ```python
    from haystack import Document
    from haystack.components.routers import MetadataRouter

    docs = [Document(content="Paris is the capital of France.", meta={"language": "en"}),
            Document(content="Berlin ist die Haupststadt von Deutschland.", meta={"language": "de"})]

    router = MetadataRouter(rules={"en": {"field": "meta.language", "operator": "==", "value": "en"}})

    print(router.run(documents=docs))

    # {'en': [Document(id=..., content: 'Paris is the capital of France.', meta: {'language': 'en'})],
    # 'unmatched': [Document(id=..., content: 'Berlin ist die Haupststadt von Deutschland.', meta: {'language': 'de'})]}
    ```
    """

    def __init__(self, rules: Dict[str, Dict]):
        """
        Initializes the MetadataRouter component.

        :param rules: A dictionary defining how to route documents to output connections based on their metadata.
            Keys are output connection names, and values are dictionaries of
            [filtering expressions](https://docs.haystack.deepset.ai/docs/metadata-filtering) in Haystack.
            For example:
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
        for rule in self.rules.values():
            if "operator" not in rule:
                raise ValueError(
                    "Invalid filter syntax. See https://docs.haystack.deepset.ai/docs/metadata-filtering for details."
                )
        component.set_output_types(self, unmatched=List[Document], **dict.fromkeys(rules, List[Document]))

    def run(self, documents: List[Document]):
        """
        Routes the documents.

        If a document does not match any of the rules, it's routed to a connection named "unmatched".

        :param documents: A list of documents to route.

        :returns: A dictionary where the keys are the names of the output connections (including `"unmatched"`)
            and the values are lists of routed documents.
        """
        unmatched_documents = []
        output: Dict[str, List[Document]] = {edge: [] for edge in self.rules}

        for document in documents:
            cur_document_matched = False
            for edge, rule in self.rules.items():
                if document_matches_filter(rule, document):
                    output[edge].append(document)
                    cur_document_matched = True

            if not cur_document_matched:
                unmatched_documents.append(document)

        output["unmatched"] = unmatched_documents
        return output
