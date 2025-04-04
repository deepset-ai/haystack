# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch
import pytest

from haystack.document_stores.in_memory.document_store import InMemoryDocumentStore
from haystack.utils.deserialization import (
    deserialize_document_store_in_init_params_inplace,
    deserialize_chatgenerator_inplace,
)
from haystack.core.errors import DeserializationError
from haystack.components.generators.chat.openai import OpenAIChatGenerator


class FakeDocumentStore:
    pass


class ChatGeneratorWithoutFromDict:
    def to_dict(self):
        return {"type": "test_deserialization.ChatGeneratorWithoutFromDict"}


class TestDeserializeDocumentStoreInInitParamsInplace:
    def test_deserialize_document_store_in_init_params_inplace(self):
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

    def test_from_dict_is_called(self):
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
                {
                    "type": "haystack.document_stores.in_memory.document_store.InMemoryDocumentStore",
                    "init_parameters": {},
                }
            )

    def test_default_from_dict_is_called(self):
        """If the document store does not provide a from_dict method, default_from_dict should be called."""
        data = {
            "type": "haystack.components.writers.document_writer.DocumentWriter",
            "init_parameters": {
                "document_store": {"type": "test_deserialization.FakeDocumentStore", "init_parameters": {}}
            },
        }

        with patch("haystack.utils.deserialization.default_from_dict") as mock_default_from_dict:
            deserialize_document_store_in_init_params_inplace(data)

            mock_default_from_dict.assert_called_once_with(
                FakeDocumentStore, {"type": "test_deserialization.FakeDocumentStore", "init_parameters": {}}
            )

    def test_missing_document_store_key(self):
        data = {"init_parameters": {"policy": "SKIP"}}
        with pytest.raises(DeserializationError):
            deserialize_document_store_in_init_params_inplace(data)

    def test_missing_type_key_in_document_store(self):
        data = {"init_parameters": {"document_store": {"init_parameters": {}}, "policy": "SKIP"}}
        with pytest.raises(DeserializationError):
            deserialize_document_store_in_init_params_inplace(data)

    def test_invalid_class_import(self):
        data = {
            "init_parameters": {
                "document_store": {"type": "invalid.module.InvalidClass", "init_parameters": {}},
                "policy": "SKIP",
            }
        }
        with pytest.raises(DeserializationError):
            deserialize_document_store_in_init_params_inplace(data)


class TestDeserializeChatGeneratorInplace:
    def test_deserialize_chatgenerator_inplace(self):
        chat_generator = OpenAIChatGenerator()
        data = {"chat_generator": chat_generator.to_dict()}

        deserialize_chatgenerator_inplace(data)
        assert isinstance(data["chat_generator"], OpenAIChatGenerator)
        assert data["chat_generator"].to_dict() == chat_generator.to_dict()

    def test_missing_chat_generator_key(self):
        data = {"some_key": "some_value"}
        with pytest.raises(DeserializationError):
            deserialize_chatgenerator_inplace(data)

    def test_chat_generator_is_not_a_dict(self):
        data = {"chat_generator": "not_a_dict"}
        with pytest.raises(DeserializationError):
            deserialize_chatgenerator_inplace(data)

    def test_type_key_missing(self):
        data = {"chat_generator": {"some_key": "some_value"}}
        with pytest.raises(DeserializationError):
            deserialize_chatgenerator_inplace(data)

    def test_class_not_correctly_imported(self):
        data = {"chat_generator": {"type": "invalid.module.InvalidClass"}}
        with pytest.raises(DeserializationError):
            deserialize_chatgenerator_inplace(data)

    def test_chat_generator_no_from_dict_method(self):
        chat_generator = ChatGeneratorWithoutFromDict()
        data = {"chat_generator": chat_generator.to_dict()}
        with pytest.raises(DeserializationError):
            deserialize_chatgenerator_inplace(data)
