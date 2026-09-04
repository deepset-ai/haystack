# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0


import pytest

from haystack.components.generators.chat.openai import OpenAIChatGenerator
from haystack.core.errors import DeserializationError
from haystack.utils.deserialization import deserialize_component_inplace


class ChatGeneratorWithoutFromDict:
    def to_dict(self):
        return {"type": "test_deserialization.ChatGeneratorWithoutFromDict"}


class TestDeserializeComponentInplace:
    def test_deserialize_component_inplace(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        chat_generator = OpenAIChatGenerator()
        data = {"chat_generator": chat_generator.to_dict()}
        deserialize_component_inplace(data)
        assert isinstance(data["chat_generator"], OpenAIChatGenerator)
        assert data["chat_generator"].to_dict() == chat_generator.to_dict()

    def test_missing_key(self):
        data = {"some_key": "some_value"}
        with pytest.raises(DeserializationError):
            deserialize_component_inplace(data)

    def test_component_is_not_a_dict(self):
        data = {"chat_generator": "not_a_dict"}
        with pytest.raises(DeserializationError):
            deserialize_component_inplace(data)

    def test_type_key_missing(self):
        data = {"chat_generator": {"some_key": "some_value"}}
        with pytest.raises(DeserializationError):
            deserialize_component_inplace(data)

    def test_class_not_correctly_imported(self):
        data = {"chat_generator": {"type": "invalid.module.InvalidClass"}}
        with pytest.raises(DeserializationError):
            deserialize_component_inplace(data)

    def test_component_no_from_dict_method(self):
        chat_generator = ChatGeneratorWithoutFromDict()
        data = {"chat_generator": chat_generator.to_dict()}
        deserialize_component_inplace(data)
        assert isinstance(data["chat_generator"], ChatGeneratorWithoutFromDict)
