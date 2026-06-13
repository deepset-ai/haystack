# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack.core.serialization import generate_qualified_class_name, import_class_by_name
from haystack.document_stores.types import DocumentStore
from haystack.tools import Tool


class ListMetadataFieldsTool(Tool):
    """Tool that lists all metadata fields and their types from a document store."""

    def __init__(self, document_store: DocumentStore) -> None:
        self.document_store = document_store
        super().__init__(
            name="list_metadata_fields",
            description=(
                "Returns all metadata fields available on documents and their types "
                "(e.g. keyword, long, date). Call this first to understand what fields "
                "you can filter on."
            ),
            parameters={"type": "object", "properties": {}},
            function=self._list_metadata_fields,
        )

    def _list_metadata_fields(self) -> dict:
        return self.document_store.get_metadata_fields_info()

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": generate_qualified_class_name(type(self)),
            "data": {"document_store": self.document_store.to_dict()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ListMetadataFieldsTool":
        store_data = data["data"]["document_store"]
        store_cls = import_class_by_name(store_data["type"])
        document_store = store_cls.from_dict(store_data)
        return cls(document_store=document_store)


class GetMetadataFieldValuesTool(Tool):
    """Tool that returns the distinct values for a given metadata field."""

    def __init__(self, document_store: DocumentStore) -> None:
        self.document_store = document_store
        super().__init__(
            name="get_metadata_field_values",
            description=(
                "Returns the distinct values present for a given metadata field. "
                "Use this to understand what values a field can take before building a filter."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "field": {"type": "string", "description": "The metadata field name."}
                },
                "required": ["field"],
            },
            function=self._get_metadata_field_values,
        )

    def _get_metadata_field_values(self, field: str) -> dict:
        values, total = self.document_store.get_metadata_field_unique_values(metadata_field=field)
        return {"values": values, "total": total}

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": generate_qualified_class_name(type(self)),
            "data": {"document_store": self.document_store.to_dict()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GetMetadataFieldValuesTool":
        store_data = data["data"]["document_store"]
        store_cls = import_class_by_name(store_data["type"])
        document_store = store_cls.from_dict(store_data)
        return cls(document_store=document_store)


class GetMetadataFieldRangeTool(Tool):
    """Tool that returns the min and max values for a numeric metadata field."""

    def __init__(self, document_store: DocumentStore) -> None:
        self.document_store = document_store
        super().__init__(
            name="get_metadata_field_range",
            description=(
                "Returns the minimum and maximum values for a numeric metadata field. "
                "Use this for fields with continuous values such as dates or counts."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "field": {"type": "string", "description": "The numeric metadata field name."}
                },
                "required": ["field"],
            },
            function=self._get_metadata_field_range,
        )

    def _get_metadata_field_range(self, field: str) -> dict:
        return self.document_store.get_metadata_field_min_max(metadata_field=field)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": generate_qualified_class_name(type(self)),
            "data": {"document_store": self.document_store.to_dict()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GetMetadataFieldRangeTool":
        store_data = data["data"]["document_store"]
        store_cls = import_class_by_name(store_data["type"])
        document_store = store_cls.from_dict(store_data)
        return cls(document_store=document_store)
