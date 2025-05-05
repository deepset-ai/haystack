import pytest
from typing import List, Dict

from haystack.dataclasses import ChatMessage, Document
from haystack.utils.state import State, _validate_schema, _schema_to_dict, _schema_from_dict


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


def test_validate_schema_valid(basic_schema):
    # Should not raise any exceptions
    _validate_schema(basic_schema)


def test_validate_schema_invalid_type():
    invalid_schema = {"test": {"type": "not_a_type"}}
    with pytest.raises(ValueError, match="must be a Python type"):
        _validate_schema(invalid_schema)


def test_validate_schema_missing_type():
    invalid_schema = {"test": {"handler": lambda x, y: x + y}}
    with pytest.raises(ValueError, match="missing a 'type' entry"):
        _validate_schema(invalid_schema)


def test_validate_schema_invalid_handler():
    invalid_schema = {"test": {"type": list, "handler": "not_callable"}}
    with pytest.raises(ValueError, match="must be callable or None"):
        _validate_schema(invalid_schema)


def test_state_initialization(basic_schema):
    # Test empty initialization
    state = State(basic_schema)
    assert state.data == {}

    # Test initialization with data
    initial_data = {"numbers": [1, 2, 3], "name": "test"}
    state = State(basic_schema, initial_data)
    assert state.data["numbers"] == [1, 2, 3]
    assert state.data["name"] == "test"


def test_state_get(basic_schema):
    state = State(basic_schema, {"name": "test"})
    assert state.get("name") == "test"
    assert state.get("non_existent") is None
    assert state.get("non_existent", "default") == "default"


def test_state_set_basic(basic_schema):
    state = State(basic_schema)

    # Test setting new values
    state.set("numbers", [1, 2])
    assert state.get("numbers") == [1, 2]

    # Test updating existing values
    state.set("numbers", [3, 4])
    assert state.get("numbers") == [1, 2, 3, 4]


def test_state_set_with_handler(complex_schema):
    state = State(complex_schema)

    # Test custom handler for numbers
    state.set("numbers", [3, 2, 1])
    assert state.get("numbers") == [1, 2, 3]

    state.set("numbers", [6, 5, 4])
    assert state.get("numbers") == [1, 2, 3, 4, 5, 6]


def test_state_set_with_handler_override(basic_schema):
    state = State(basic_schema)

    # Custom handler that concatenates strings
    custom_handler = lambda current, new: f"{current}-{new}" if current else new

    state.set("name", "first")
    state.set("name", "second", handler_override=custom_handler)
    assert state.get("name") == "first-second"


def test_state_has(basic_schema):
    state = State(basic_schema, {"name": "test"})
    assert state.has("name") is True
    assert state.has("non_existent") is False


def test_state_empty_schema():
    state = State({})
    assert state.data == {}

    # Instead of comparing the entire schema directly, check structure separately
    assert "messages" in state.schema
    assert state.schema["messages"]["type"] == List[ChatMessage]
    assert callable(state.schema["messages"]["handler"])

    with pytest.raises(ValueError, match="Key 'any_key' not found in schema"):
        state.set("any_key", "value")


def test_state_none_values(basic_schema):
    state = State(basic_schema)
    state.set("name", None)
    assert state.get("name") is None
    state.set("name", "value")
    assert state.get("name") == "value"


def test_state_merge_lists(basic_schema):
    state = State(basic_schema)
    state.set("numbers", "not_a_list")
    assert state.get("numbers") == ["not_a_list"]
    state.set("numbers", [1, 2])
    assert state.get("numbers") == ["not_a_list", 1, 2]


def test_state_nested_structures():
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


def test_schema_to_dict(basic_schema):
    expected_dict = {"numbers": {"type": "list"}, "metadata": {"type": "dict"}, "name": {"type": "str"}}
    result = _schema_to_dict(basic_schema)
    assert result == expected_dict


def test_schema_to_dict_with_handlers(complex_schema):
    expected_dict = {
        "numbers": {"type": "list", "handler": "test_state.numbers_handler"},
        "metadata": {"type": "dict"},
        "name": {"type": "str"},
    }
    result = _schema_to_dict(complex_schema)
    assert result == expected_dict


def test_schema_from_dict(basic_schema):
    schema_dict = {"numbers": {"type": "list"}, "metadata": {"type": "dict"}, "name": {"type": "str"}}
    result = _schema_from_dict(schema_dict)
    assert result == basic_schema


def test_schema_from_dict_with_handlers(complex_schema):
    schema_dict = {
        "numbers": {"type": "list", "handler": "test_state.numbers_handler"},
        "metadata": {"type": "dict"},
        "name": {"type": "str"},
    }
    result = _schema_from_dict(schema_dict)
    assert result == complex_schema


def test_state_mutability():
    state = State({"my_list": {"type": list}}, {"my_list": [1, 2]})

    my_list = state.get("my_list")
    my_list.append(3)

    assert state.get("my_list") == [1, 2]


def test_state_to_dict():
    # we test dict, a python type and a haystack dataclass
    state_schema = {"numbers": {"type": int}, "messages": {"type": List[ChatMessage]}, "dict_of_lists": {"type": dict}}

    data = {
        "numbers": 1,
        "messages": [ChatMessage.from_user(text="Hello, world!")],
        "dict_of_lists": {"numbers": [1, 2, 3]},
    }
    state = State(state_schema, data)
    state_dict = state.to_dict()
    assert state_dict["schema"] == {
        "numbers": {"type": "int", "handler": "haystack.dataclasses.state_utils.replace_values"},
        "messages": {
            "type": "typing.List[haystack.dataclasses.chat_message.ChatMessage]",
            "handler": "haystack.dataclasses.state_utils.merge_lists",
        },
        "dict_of_lists": {"type": "dict", "handler": "haystack.dataclasses.state_utils.replace_values"},
    }
    assert state_dict["data"] == {
        "numbers": 1,
        "messages": [
            {"role": "user", "meta": {}, "name": None, "content": [{"text": "Hello, world!"}], "_type": "ChatMessage"}
        ],
        "dict_of_lists": {"numbers": [1, 2, 3]},
    }


def test_state_from_dict():
    state_dict = {
        "schema": {
            "numbers": {"type": "int", "handler": "haystack.dataclasses.state_utils.replace_values"},
            "messages": {
                "type": "typing.List[haystack.dataclasses.chat_message.ChatMessage]",
                "handler": "haystack.dataclasses.state_utils.merge_lists",
            },
            "dict_of_lists": {"type": "dict", "handler": "haystack.dataclasses.state_utils.replace_values"},
        },
        "data": {
            "numbers": 1,
            "messages": [
                {
                    "role": "user",
                    "meta": {},
                    "name": None,
                    "content": [{"text": "Hello, world!"}],
                    "_type": "ChatMessage",
                }
            ],
            "dict_of_lists": {"numbers": [1, 2, 3]},
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
