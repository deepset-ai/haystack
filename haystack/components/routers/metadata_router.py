# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Type, Union

from haystack import Document, component
from haystack.dataclasses import ByteStream
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

    def __init__(self, rules: Dict[str, Dict], output_type: Type = List[Document]) -> None:
        """
        Initializes the MetadataRouter component.

        :param rules: A dictionary defining how to route documents or byte streams to output connections based on their
            metadata. Keys are output connection names, and values are dictionaries of
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
        self.output_type = output_type
        component.set_output_types(self, unmatched=self.output_type, **dict.fromkeys(rules, self.output_type))

    def run(
        self, documents: Union[List[Document], List[ByteStream]]
    ) -> Dict[str, Union[List[Document], List[ByteStream]]]:
        """
        Routes documents or byte streams to different connections based on their metadata fields.

        If a document or byte stream does not match any of the rules, it's routed to a connection named "unmatched".

        :param documents: A list of `Document` or `ByteStream` objects to be routed based on their metadata.

        :returns: A dictionary where the keys are the names of the output connections (including `"unmatched"`)
            and the values are lists of `Document` or `ByteStream` objects that matched the corresponding rules.
        """
        unmatched = []
        output: Dict[str, Union[List[Document], List[ByteStream]]] = {edge: [] for edge in self.rules}

        for doc_or_bytestream in documents:
            current_obj_matched = False
            for edge, rule in self.rules.items():
                if document_matches_filter(filters=rule, document=doc_or_bytestream):
                    output[edge].append(doc_or_bytestream)
                    current_obj_matched = True

            if not current_obj_matched:
                unmatched.append(doc_or_bytestream)

        output["unmatched"] = unmatched
        return output
