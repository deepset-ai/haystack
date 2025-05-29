# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from dataclasses import dataclass

from haystack.dataclasses import ChatMessage
from haystack.components.agents.state.state import (
    State,
    _validate_schema,
    _schema_to_dict,
    _schema_from_dict,
    _is_list_type,
    merge_lists,
    _is_valid_type,
)
from typing import List, Dict, Optional, Union, TypeVar, Generic
import inspect


@pytest.fixture
def basic_schema():
    return {"numbers": {"type": list}, "metadata": {"type": dict}, "name": {"type": str}}


def numbers_handler(current, new):
    if current is None:
        return sorted(set(new))
    return sorted(set(current + new))


@pytest.fixture
def complex_schema():
    return {"numbers": {"type": list, "handler": numbers_handler}, "metadata": {"type": dict}, "name": {"type": str}}


def test_is_list_type():
    assert _is_list_type(list) is True
    assert _is_list_type(List[int]) is True
    assert _is_list_type(List[str]) is True
    assert _is_list_type(dict) is False
    assert _is_list_type(int) is False
    assert _is_list_type(Union[List[int], None]) is False


class TestMergeLists:
    def test_merge_two_lists(self):
        current = [1, 2, 3]
        new = [4, 5, 6]
        result = merge_lists(current, new)
        assert result == [1, 2, 3, 4, 5, 6]
        # Ensure original lists weren't modified
        assert current == [1, 2, 3]
        assert new == [4, 5, 6]

    def test_append_to_list(self):
        current = [1, 2, 3]
        new = 4
        result = merge_lists(current, new)
        assert result == [1, 2, 3, 4]
        assert current == [1, 2, 3]  # Ensure original wasn't modified

    def test_create_new_list(self):
        current = 1
        new = 2
        result = merge_lists(current, new)
        assert result == [1, 2]

    def test_replace_with_list(self):
        current = 1
        new = [2, 3]
        result = merge_lists(current, new)
        assert result == [1, 2, 3]


class TestIsValidType:
    def test_builtin_types(self):
        assert _is_valid_type(str) is True
        assert _is_valid_type(int) is True
        assert _is_valid_type(dict) is True
        assert _is_valid_type(list) is True
        assert _is_valid_type(tuple) is True
        assert _is_valid_type(set) is True
        assert _is_valid_type(bool) is True
        assert _is_valid_type(float) is True

    def test_generic_types(self):
        assert _is_valid_type(List[str]) is True
        assert _is_valid_type(Dict[str, int]) is True
        assert _is_valid_type(List[Dict[str, int]]) is True
        assert _is_valid_type(Dict[str, List[int]]) is True

    def test_custom_classes(self):
        @dataclass
        class CustomClass:
            value: int

        T = TypeVar("T")

        class GenericCustomClass(Generic[T]):
            pass

        # Test regular and generic custom classes
        assert _is_valid_type(CustomClass) is True
        assert _is_valid_type(GenericCustomClass) is True
        assert _is_valid_type(GenericCustomClass[int]) is True

        # Test generic types with custom classes
        assert _is_valid_type(List[CustomClass]) is True
        assert _is_valid_type(Dict[str, CustomClass]) is True
        assert _is_valid_type(Dict[str, GenericCustomClass[int]]) is True

    def test_invalid_types(self):
        # Test regular values
        assert _is_valid_type(42) is False
        assert _is_valid_type("string") is False
        assert _is_valid_type([1, 2, 3]) is False
        assert _is_valid_type({"a": 1}) is False
        assert _is_valid_type(True) is False

        # Test class instances
        @dataclass
        class SampleClass:
            value: int

        instance = SampleClass(42)
        assert _is_valid_type(instance) is False

        # Test callable objects
        assert _is_valid_type(len) is False
        assert _is_valid_type(lambda x: x) is False
        assert _is_valid_type(print) is False

    def test_union_and_optional_types(self):
        # Test basic Union types
        assert _is_valid_type(Union[str, int]) is True
        assert _is_valid_type(Union[str, None]) is True
        assert _is_valid_type(Union[List[int], Dict[str, str]]) is True

        # Test Optional types (which are Union[T, None])
        assert _is_valid_type(Optional[str]) is True
        assert _is_valid_type(Optional[List[int]]) is True
        assert _is_valid_type(Optional[Dict[str, list]]) is True

        # Test that Union itself is not a valid type (only instantiated Unions are)
        assert _is_valid_type(Union) is False

    def test_nested_generic_types(self):
        assert _is_valid_type(List[List[Dict[str, List[int]]]]) is True
        assert _is_valid_type(Dict[str, List[Dict[str, set]]]) is True
        assert _is_valid_type(Dict[str, Optional[List[int]]]) is True
        assert _is_valid_type(List[Union[str, Dict[str, List[int]]]]) is True

    def test_edge_cases(self):
        # Test None and NoneType
        assert _is_valid_type(None) is False
        assert _is_valid_type(type(None)) is True

        # Test functions and methods
        def sample_func():
            pass

        assert _is_valid_type(sample_func) is False
        assert _is_valid_type(type(sample_func)) is True

        # Test modules
        assert _is_valid_type(inspect) is False

        # Test type itself
        assert _is_valid_type(type) is True

    @pytest.mark.parametrize(
        "test_input,expected",
        [
            (str, True),
            (int, True),
            (List[int], True),
            (Dict[str, int], True),
            (Union[str, int], True),
            (Optional[str], True),
            (42, False),
            ("string", False),
            ([1, 2, 3], False),
            (lambda x: x, False),
        ],
    )
    def test_parametrized_cases(self, test_input, expected):
        assert _is_valid_type(test_input) is expected


class TestState:
    def test_validate_schema_valid(self, basic_schema):
        # Should not raise any exceptions
        _validate_schema(basic_schema)

    def test_validate_schema_invalid_type(self):
        invalid_schema = {"test": {"type": "not_a_type"}}
        with pytest.raises(ValueError, match="must be a Python type"):
            _validate_schema(invalid_schema)

    def test_validate_schema_missing_type(self):
        invalid_schema = {"test": {"handler": lambda x, y: x + y}}
        with pytest.raises(ValueError, match="missing a 'type' entry"):
            _validate_schema(invalid_schema)

    def test_validate_schema_invalid_handler(self):
        invalid_schema = {"test": {"type": list, "handler": "not_callable"}}
        with pytest.raises(ValueError, match="must be callable or None"):
            _validate_schema(invalid_schema)

    def test_state_initialization(self, basic_schema):
        # Test empty initialization
        state = State(basic_schema)
        assert state.data == {}

        # Test initialization with data
        initial_data = {"numbers": [1, 2, 3], "name": "test"}
        state = State(basic_schema, initial_data)
        assert state.data["numbers"] == [1, 2, 3]
        assert state.data["name"] == "test"

    def test_state_get(self, basic_schema):
        state = State(basic_schema, {"name": "test"})
        assert state.get("name") == "test"
        assert state.get("non_existent") is None
        assert state.get("non_existent", "default") == "default"

    def test_state_set_basic(self, basic_schema):
        state = State(basic_schema)

        # Test setting new values
        state.set("numbers", [1, 2])
        assert state.get("numbers") == [1, 2]

        # Test updating existing values
        state.set("numbers", [3, 4])
        assert state.get("numbers") == [1, 2, 3, 4]

    def test_state_set_with_handler(self, complex_schema):
        state = State(complex_schema)

        # Test custom handler for numbers
        state.set("numbers", [3, 2, 1])
        assert state.get("numbers") == [1, 2, 3]

        state.set("numbers", [6, 5, 4])
        assert state.get("numbers") == [1, 2, 3, 4, 5, 6]

    def test_state_set_with_handler_override(self, basic_schema):
        state = State(basic_schema)

        # Custom handler that concatenates strings
        custom_handler = lambda current, new: f"{current}-{new}" if current else new

        state.set("name", "first")
        state.set("name", "second", handler_override=custom_handler)
        assert state.get("name") == "first-second"

    def test_state_has(self, basic_schema):
        state = State(basic_schema, {"name": "test"})
        assert state.has("name") is True
        assert state.has("non_existent") is False

    def test_state_empty_schema(self):
        state = State({})
        assert state.data == {}

        # Instead of comparing the entire schema directly, check structure separately
        assert "messages" in state.schema
        assert state.schema["messages"]["type"] == List[ChatMessage]
        assert callable(state.schema["messages"]["handler"])

        with pytest.raises(ValueError, match="Key 'any_key' not found in schema"):
            state.set("any_key", "value")

    def test_state_none_values(self, basic_schema):
        state = State(basic_schema)
        state.set("name", None)
        assert state.get("name") is None
        state.set("name", "value")
        assert state.get("name") == "value"

    def test_state_merge_lists(self, basic_schema):
        state = State(basic_schema)
        state.set("numbers", "not_a_list")
        assert state.get("numbers") == ["not_a_list"]
        state.set("numbers", [1, 2])
        assert state.get("numbers") == ["not_a_list", 1, 2]

    def test_state_nested_structures(self):
        schema = {
            "complex": {
                "type": Dict[str, List[int]],
                "handler": lambda current, new: {
                    k: current.get(k, []) + new.get(k, []) for k in set(current.keys()) | set(new.keys())
                }
                if current
                else new,
            }
        }

        state = State(schema)
        state.set("complex", {"a": [1, 2], "b": [3, 4]})
        state.set("complex", {"b": [5, 6], "c": [7, 8]})

        expected = {"a": [1, 2], "b": [3, 4, 5, 6], "c": [7, 8]}
        assert state.get("complex") == expected

    def test_schema_to_dict(self, basic_schema):
        expected_dict = {"numbers": {"type": "list"}, "metadata": {"type": "dict"}, "name": {"type": "str"}}
        result = _schema_to_dict(basic_schema)
        assert result == expected_dict

    def test_schema_to_dict_with_handlers(self, complex_schema):
        expected_dict = {
            "numbers": {"type": "list", "handler": "test_state_class.numbers_handler"},
            "metadata": {"type": "dict"},
            "name": {"type": "str"},
        }
        result = _schema_to_dict(complex_schema)
        assert result == expected_dict

    def test_schema_from_dict(self, basic_schema):
        schema_dict = {"numbers": {"type": "list"}, "metadata": {"type": "dict"}, "name": {"type": "str"}}
        result = _schema_from_dict(schema_dict)
        assert result == basic_schema

    def test_schema_from_dict_with_handlers(self, complex_schema):
        schema_dict = {
            "numbers": {"type": "list", "handler": "test_state_class.numbers_handler"},
            "metadata": {"type": "dict"},
            "name": {"type": "str"},
        }
        result = _schema_from_dict(schema_dict)
        assert result == complex_schema

    def test_state_mutability(self):
        state = State({"my_list": {"type": list}}, {"my_list": [1, 2]})

        my_list = state.get("my_list")
        my_list.append(3)

        assert state.get("my_list") == [1, 2]

    def test_state_to_dict(self):
        # we test dict, a python type and a haystack dataclass
        state_schema = {
            "numbers": {"type": int},
            "messages": {"type": List[ChatMessage]},
            "dict_of_lists": {"type": dict},
        }

        data = {
            "numbers": 1,
            "messages": [ChatMessage.from_user(text="Hello, world!")],
            "dict_of_lists": {"numbers": [1, 2, 3]},
        }
        state = State(state_schema, data)
        state_dict = state.to_dict()
        assert state_dict["schema"] == {
            "numbers": {"type": "int", "handler": "haystack.components.agents.state.state_utils.replace_values"},
            "messages": {
                "type": "typing.List[haystack.dataclasses.chat_message.ChatMessage]",
                "handler": "haystack.components.agents.state.state_utils.merge_lists",
            },
            "dict_of_lists": {"type": "dict", "handler": "haystack.components.agents.state.state_utils.replace_values"},
        }
        assert state_dict["data"] == {
            "serialization_schema": {
                "numbers": {"type": "integer"},
                "messages": {"type": "array", "items": {"type": "haystack.dataclasses.chat_message.ChatMessage"}},
                "dict_of_lists": {"type": "object"},
            },
            "serialized_data": {
                "numbers": 1,
                "messages": [{"role": "user", "meta": {}, "name": None, "content": [{"text": "Hello, world!"}]}],
                "dict_of_lists": {"numbers": [1, 2, 3]},
            },
        }

    def test_state_from_dict(self):
        state_dict = {
            "schema": {
                "numbers": {"type": "int", "handler": "haystack.components.agents.state.state_utils.replace_values"},
                "messages": {
                    "type": "typing.List[haystack.dataclasses.chat_message.ChatMessage]",
                    "handler": "haystack.components.agents.state.state_utils.merge_lists",
                },
                "dict_of_lists": {
                    "type": "dict",
                    "handler": "haystack.components.agents.state.state_utils.replace_values",
                },
            },
            "data": {
                "serialization_schema": {
                    "numbers": {"type": "integer"},
                    "messages": {"type": "array", "items": {"type": "haystack.dataclasses.chat_message.ChatMessage"}},
                    "dict_of_lists": {"type": "object"},
                },
                "serialized_data": {
                    "numbers": 1,
                    "messages": [{"role": "user", "meta": {}, "name": None, "content": [{"text": "Hello, world!"}]}],
                    "dict_of_lists": {"numbers": [1, 2, 3]},
                },
            },
        }
        state = State.from_dict(state_dict)
        # Check types are correctly converted
        assert state.schema["numbers"]["type"] == int
        assert state.schema["dict_of_lists"]["type"] == dict
        # Check handlers are functions, not comparing exact functions as they might be different references
        assert callable(state.schema["numbers"]["handler"])
        assert callable(state.schema["messages"]["handler"])
        assert callable(state.schema["dict_of_lists"]["handler"])
        # Check data is correct
        assert state.data["numbers"] == 1
        assert state.data["messages"] == [ChatMessage.from_user(text="Hello, world!")]
        assert state.data["dict_of_lists"] == {"numbers": [1, 2, 3]}
