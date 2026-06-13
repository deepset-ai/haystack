# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

from haystack.dataclasses import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.tools.document_store_tools import (
    GetMetadataFieldRangeTool,
    GetMetadataFieldValuesTool,
    ListMetadataFieldsTool,
)


@pytest.fixture()
def populated_store():
    store = InMemoryDocumentStore()
    store.write_documents(
        [
            Document(content="a", meta={"genre": "fiction", "year": 2020}),
            Document(content="b", meta={"genre": "fiction", "year": 2021}),
            Document(content="c", meta={"genre": "non-fiction", "year": 2019}),
        ]
    )
    return store


class TestListMetadataFieldsTool:
    def test_init(self, populated_store):
        tool = ListMetadataFieldsTool(populated_store)
        assert tool.name == "list_metadata_fields"
        assert "filter" in tool.description
        assert tool.parameters == {"type": "object", "properties": {}}

    def test_invoke(self, populated_store):
        tool = ListMetadataFieldsTool(populated_store)
        result = tool.invoke()
        assert isinstance(result, dict)
        assert "genre" in result
        assert "year" in result

    def test_to_dict(self, populated_store):
        tool = ListMetadataFieldsTool(populated_store)
        d = tool.to_dict()
        assert d["type"] == "haystack.tools.document_store_tools.ListMetadataFieldsTool"
        assert "document_store" in d["data"]
        assert d["data"]["document_store"]["type"] == "haystack.document_stores.in_memory.document_store.InMemoryDocumentStore"

    def test_from_dict_roundtrip(self, populated_store):
        tool = ListMetadataFieldsTool(populated_store)
        restored = ListMetadataFieldsTool.from_dict(tool.to_dict())
        assert restored.name == tool.name
        assert isinstance(restored.document_store, InMemoryDocumentStore)


class TestGetMetadataFieldValuesTool:
    def test_init(self, populated_store):
        tool = GetMetadataFieldValuesTool(populated_store)
        assert tool.name == "get_metadata_field_values"
        assert tool.parameters["required"] == ["field"]

    def test_invoke_returns_dict_with_values_and_total(self, populated_store):
        tool = GetMetadataFieldValuesTool(populated_store)
        result = tool.invoke(field="genre")
        assert "values" in result
        assert "total" in result
        assert set(result["values"]) == {"fiction", "non-fiction"}
        assert result["total"] == 2

    def test_to_dict(self, populated_store):
        tool = GetMetadataFieldValuesTool(populated_store)
        d = tool.to_dict()
        assert d["type"] == "haystack.tools.document_store_tools.GetMetadataFieldValuesTool"

    def test_from_dict_roundtrip(self, populated_store):
        tool = GetMetadataFieldValuesTool(populated_store)
        restored = GetMetadataFieldValuesTool.from_dict(tool.to_dict())
        assert restored.name == tool.name
        assert isinstance(restored.document_store, InMemoryDocumentStore)


class TestGetMetadataFieldRangeTool:
    def test_init(self, populated_store):
        tool = GetMetadataFieldRangeTool(populated_store)
        assert tool.name == "get_metadata_field_range"
        assert tool.parameters["required"] == ["field"]

    def test_invoke(self, populated_store):
        tool = GetMetadataFieldRangeTool(populated_store)
        result = tool.invoke(field="year")
        assert result["min"] == 2019
        assert result["max"] == 2021

    def test_to_dict(self, populated_store):
        tool = GetMetadataFieldRangeTool(populated_store)
        d = tool.to_dict()
        assert d["type"] == "haystack.tools.document_store_tools.GetMetadataFieldRangeTool"

    def test_from_dict_roundtrip(self, populated_store):
        tool = GetMetadataFieldRangeTool(populated_store)
        restored = GetMetadataFieldRangeTool.from_dict(tool.to_dict())
        assert restored.name == tool.name
        assert isinstance(restored.document_store, InMemoryDocumentStore)


class TestPublicExports:
    def test_importable_from_haystack_tools(self):
        from haystack.tools import GetMetadataFieldRangeTool, GetMetadataFieldValuesTool, ListMetadataFieldsTool

        assert ListMetadataFieldsTool is not None
        assert GetMetadataFieldValuesTool is not None
        assert GetMetadataFieldRangeTool is not None
