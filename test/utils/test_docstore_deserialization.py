# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch
import pytest

from haystack.document_stores.in_memory.document_store import InMemoryDocumentStore
from haystack.utils.docstore_deserialization import deserialize_document_store_in_init_params_inplace
from haystack.core.errors import DeserializationError


class FakeDocumentStore:
    pass


def test_deserialize_document_store_in_init_params_inplace():
    data = {
        "type": "haystack.components.writers.document_writer.DocumentWriter",
        "init_parameters": {
            "document_store": {
                "type": "haystack.document_stores.in_memory.document_store.InMemoryDocumentStore",
                "init_parameters": {},
            }
        },
    }

    deserialize_document_store_in_init_params_inplace(data)
    assert isinstance(data["init_parameters"]["document_store"], InMemoryDocumentStore)


def test_from_dict_is_called():
    """If the document store provides a from_dict method, it should be called."""
    data = {
        "type": "haystack.components.writers.document_writer.DocumentWriter",
        "init_parameters": {
            "document_store": {
                "type": "haystack.document_stores.in_memory.document_store.InMemoryDocumentStore",
                "init_parameters": {},
            }
        },
    }

    with patch.object(InMemoryDocumentStore, "from_dict") as mock_from_dict:
        deserialize_document_store_in_init_params_inplace(data)

        mock_from_dict.assert_called_once_with(
            {"type": "haystack.document_stores.in_memory.document_store.InMemoryDocumentStore", "init_parameters": {}}
        )


def test_default_from_dict_is_called():
    """If the document store does not provide a from_dict method, default_from_dict should be called."""
    data = {
        "type": "haystack.components.writers.document_writer.DocumentWriter",
        "init_parameters": {
            "document_store": {"type": "test_docstore_deserialization.FakeDocumentStore", "init_parameters": {}}
        },
    }

    with patch("haystack.utils.docstore_deserialization.default_from_dict") as mock_default_from_dict:
        deserialize_document_store_in_init_params_inplace(data)

        mock_default_from_dict.assert_called_once_with(
            FakeDocumentStore, {"type": "test_docstore_deserialization.FakeDocumentStore", "init_parameters": {}}
        )


def test_missing_document_store_key():
    data = {"init_parameters": {"policy": "SKIP"}}
    with pytest.raises(DeserializationError):
        deserialize_document_store_in_init_params_inplace(data)


def test_missing_type_key_in_document_store():
    data = {"init_parameters": {"document_store": {"init_parameters": {}}, "policy": "SKIP"}}
    with pytest.raises(DeserializationError):
        deserialize_document_store_in_init_params_inplace(data)


def test_invalid_class_import():
    data = {
        "init_parameters": {
            "document_store": {"type": "invalid.module.InvalidClass", "init_parameters": {}},
            "policy": "SKIP",
        }
    }
    with pytest.raises(DeserializationError):
        deserialize_document_store_in_init_params_inplace(data)
