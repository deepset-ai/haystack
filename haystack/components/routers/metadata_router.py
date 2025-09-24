# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Union

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.dataclasses import ByteStream
from haystack.utils import deserialize_type, serialize_type
from haystack.utils.filters import document_matches_filter


@component
class MetadataRouter:
    """
    Routes documents or byte streams to different connections based on their metadata fields.

    Specify the routing rules in the `init` method.
    If a document or byte stream does not match any of the rules, it's routed to a connection named "unmatched".


    ### Usage examples

    **Routing Documents by metadata:**
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

    **Routing ByteStreams by metadata:**
    ```python
    from haystack.dataclasses import ByteStream
    from haystack.components.routers import MetadataRouter

    streams = [
        ByteStream.from_string("Hello world", meta={"language": "en"}),
        ByteStream.from_string("Bonjour le monde", meta={"language": "fr"})
    ]

    router = MetadataRouter(
        rules={"english": {"field": "meta.language", "operator": "==", "value": "en"}},
        output_type=list[ByteStream]
    )

    result = router.run(documents=streams)
    # {'english': [ByteStream(...)], 'unmatched': [ByteStream(...)]}
    ```
    """

    def __init__(self, rules: dict[str, dict], output_type: type = list[Document]) -> None:
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
            :param output_type: The type of the output produced. Lists of Documents or ByteStreams can be specified.
        """
        self.rules = rules
        self.output_type = output_type
        for rule in self.rules.values():
            if "operator" not in rule:
                raise ValueError(
                    "Invalid filter syntax. See https://docs.haystack.deepset.ai/docs/metadata-filtering for details."
                )
        component.set_output_types(self, unmatched=self.output_type, **dict.fromkeys(rules, self.output_type))

    def run(self, documents: Union[list[Document], list[ByteStream]]):
        """
        Routes documents or byte streams to different connections based on their metadata fields.

        If a document or byte stream does not match any of the rules, it's routed to a connection named "unmatched".

        :param documents: A list of `Document` or `ByteStream` objects to be routed based on their metadata.

        :returns: A dictionary where the keys are the names of the output connections (including `"unmatched"`)
            and the values are lists of `Document` or `ByteStream` objects that matched the corresponding rules.
        """

        unmatched: Union[list[Document], list[ByteStream]] = []
        output: dict[str, Union[list[Document], list[ByteStream]]] = {edge: [] for edge in self.rules}

        for doc_or_bytestream in documents:
            current_obj_matched = False
            for edge, rule in self.rules.items():
                if document_matches_filter(filters=rule, document=doc_or_bytestream):
                    # we need to ignore the arg-type here because the underlying
                    # filter methods use type Union[Document, ByteStream]
                    output[edge].append(doc_or_bytestream)  # type: ignore[arg-type]
                    current_obj_matched = True

            if not current_obj_matched:
                unmatched.append(doc_or_bytestream)  # type: ignore[arg-type]

        output["unmatched"] = unmatched
        return output

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :returns:
            The serialized component as a dictionary.
        """
        return default_to_dict(self, rules=self.rules, output_type=serialize_type(self.output_type))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MetadataRouter":
        """
        Deserialize this component from a dictionary.

        :param data:
            The dictionary representation of this component.
        :returns:
            The deserialized component instance.
        """
        init_params = data.get("init_parameters", {})
        if "output_type" in init_params:
            # Deserialize the output_type to its original type
            init_params["output_type"] = deserialize_type(init_params["output_type"])
        return default_from_dict(cls, data)
