# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack.components.agents.state import LsStateTool, ReadStateTool, StateToolset, WriteStateTool
from haystack.components.agents.state.state import State
from haystack.components.agents.state.state_tools import _ls_state, _read_state, _write_state
from haystack.components.tools.tool_invoker import ToolInvoker
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.tools import Tool, Toolset
from haystack.tools.from_function import create_tool_from_function


@pytest.fixture
def state():
    return State(
        schema={"name": {"type": str}, "count": {"type": int}, "tags": {"type": list}},
        data={"name": "Alice", "count": 42},
    )


class TestLsStateFunction:
    def test_lists_keys_and_types(self, state):
        result = _ls_state(state)
        assert "- name (str)" in result
        assert "- count (int)" in result
        assert "- tags (list)" in result
        assert "- messages (list)" in result  # always present

    def test_empty_schema_returns_only_messages(self):
        result = _ls_state(State(schema={}))
        assert result == "- messages (list)"


class TestCatStateFunction:
    def test_reads_existing_key(self, state):
        result = _read_state("name", state)
        assert result == repr("Alice")

    def test_missing_key_returns_error_message(self, state):
        result = _read_state("missing", state)
        assert "not found" in result
        assert "ls_state" in result

    def test_truncates_long_value_by_default(self):
        long_value = "x" * 500
        state = State(schema={"data": {"type": str}}, data={"data": long_value})
        result = _read_state("data", state)
        assert "truncated" in result
        assert len(result) < len(repr(long_value))

    def test_no_truncation_when_disabled(self):
        long_value = "x" * 500
        state = State(schema={"data": {"type": str}}, data={"data": long_value})
        result = _read_state("data", state, truncate=False)
        assert "truncated" not in result
        assert repr(long_value) in result

    def test_short_value_not_truncated(self, state):
        result = _read_state("name", state)
        assert "truncated" not in result


class TestWriteStateFunction:
    def test_writes_string_key(self):
        state = State(schema={"answer": {"type": str}})
        result = _write_state("answer", "hello", state)
        assert "updated successfully" in result
        assert state.get("answer") == "hello"

    def test_missing_key_returns_error_message(self):
        state = State(schema={"answer": {"type": str}})
        result = _write_state("missing", "value", state)
        assert "not found" in result
        assert "ls_state" in result

    def test_non_string_key_returns_error_message(self):
        state = State(schema={"count": {"type": int}})
        result = _write_state("count", "42", state)
        assert "int" in result
        assert "write_state only supports string-typed keys" in result

    def test_does_not_modify_state_on_error(self):
        state = State(schema={"count": {"type": int}})
        _write_state("count", "42", state)
        assert not state.has("count")


class TestLsStateTool:
    def test_defaults(self):
        tool = LsStateTool()
        assert tool.name == "ls_state"
        assert "List" in tool.description
        assert tool.parameters == {"type": "object", "properties": {}}
        assert tool.function is _ls_state

    def test_custom_name_and_description(self):
        tool = LsStateTool(name="list_keys", description="Show keys.")
        assert tool.name == "list_keys"
        assert tool.description == "Show keys."

    def test_is_tool_instance(self):
        assert isinstance(LsStateTool(), Tool)

    def test_to_dict(self):
        tool = LsStateTool(name="list_keys", description="Show keys.")
        d = tool.to_dict()
        assert d["type"].endswith("LsStateTool")
        assert d["data"] == {"name": "list_keys", "description": "Show keys."}

    def test_from_dict_round_trip(self):
        tool = LsStateTool(name="list_keys", description="Show keys.")
        restored = LsStateTool.from_dict(tool.to_dict())
        assert restored.name == tool.name
        assert restored.description == tool.description
        assert restored.parameters == tool.parameters
        assert restored.function is _ls_state

    def test_from_dict_defaults(self):
        original = LsStateTool()
        restored = LsStateTool.from_dict(original.to_dict())
        assert restored.name == original.name
        assert restored.description == original.description


class TestReadStateTool:
    def test_defaults(self):
        tool = ReadStateTool()
        assert tool.name == "read_state"
        assert tool.parameters["type"] == "object"
        assert "key" in tool.parameters["properties"]
        assert "truncate" in tool.parameters["properties"]
        assert tool.parameters["required"] == ["key"]
        assert tool.function is _read_state

    def test_custom_descriptions(self):
        tool = ReadStateTool(
            name="read_key", description="Read a key.", key_description="Which key?", truncate_description="Truncate?"
        )
        assert tool.name == "read_key"
        assert tool.parameters["properties"]["key"]["description"] == "Which key?"
        assert tool.parameters["properties"]["truncate"]["description"] == "Truncate?"

    def test_is_tool_instance(self):
        assert isinstance(ReadStateTool(), Tool)

    def test_to_dict(self):
        tool = ReadStateTool(name="read_key", description="Read.", key_description="k", truncate_description="t")
        d = tool.to_dict()
        assert d["type"].endswith("ReadStateTool")
        assert d["data"]["name"] == "read_key"
        assert d["data"]["key_description"] == "k"
        assert d["data"]["truncate_description"] == "t"

    def test_from_dict_round_trip(self):
        tool = ReadStateTool(name="read_key", description="Read.", key_description="k", truncate_description="t")
        restored = ReadStateTool.from_dict(tool.to_dict())
        assert restored.name == tool.name
        assert restored.description == tool.description
        assert restored.key_description == tool.key_description
        assert restored.truncate_description == tool.truncate_description
        assert restored.parameters == tool.parameters
        assert restored.function is _read_state

    def test_from_dict_defaults(self):
        original = ReadStateTool()
        restored = ReadStateTool.from_dict(original.to_dict())
        assert restored.name == original.name
        assert restored.key_description == original.key_description


class TestWriteStateTool:
    def test_defaults(self):
        tool = WriteStateTool()
        assert tool.name == "write_state"
        assert "key" in tool.parameters["properties"]
        assert "value" in tool.parameters["properties"]
        assert set(tool.parameters["required"]) == {"key", "value"}
        assert tool.function is _write_state

    def test_custom_descriptions(self):
        tool = WriteStateTool(
            name="set_key", description="Set a key.", key_description="Which key?", value_description="What value?"
        )
        assert tool.name == "set_key"
        assert tool.parameters["properties"]["key"]["description"] == "Which key?"
        assert tool.parameters["properties"]["value"]["description"] == "What value?"

    def test_is_tool_instance(self):
        assert isinstance(WriteStateTool(), Tool)

    def test_to_dict(self):
        tool = WriteStateTool(name="set_key", description="Set.", key_description="k", value_description="v")
        d = tool.to_dict()
        assert d["type"].endswith("WriteStateTool")
        assert d["data"]["name"] == "set_key"
        assert d["data"]["key_description"] == "k"
        assert d["data"]["value_description"] == "v"

    def test_from_dict_round_trip(self):
        tool = WriteStateTool(name="set_key", description="Set.", key_description="k", value_description="v")
        restored = WriteStateTool.from_dict(tool.to_dict())
        assert restored.name == tool.name
        assert restored.description == tool.description
        assert restored.key_description == tool.key_description
        assert restored.value_description == tool.value_description
        assert restored.parameters == tool.parameters
        assert restored.function is _write_state

    def test_from_dict_defaults(self):
        original = WriteStateTool()
        restored = WriteStateTool.from_dict(original.to_dict())
        assert restored.name == original.name
        assert restored.value_description == original.value_description


class TestStateToolset:
    def test_is_toolset_instance(self):
        assert isinstance(StateToolset, Toolset)

    def test_contains_all_three_tools(self):
        names = {tool.name for tool in StateToolset.tools}
        assert names == {"ls_state", "read_state", "write_state"}

    def test_tool_types(self):
        tool_types = {type(t) for t in StateToolset.tools}
        assert tool_types == {LsStateTool, ReadStateTool, WriteStateTool}


class TestStateParameterExcludedFromSchema:
    def test_state_param_not_in_schema(self):
        def my_tool(query: str, state: State) -> str:
            return state.get("data") or query

        tool = create_tool_from_function(my_tool)
        assert "query" in tool.parameters["properties"]
        assert "state" not in tool.parameters["properties"]

    def test_only_state_param_produces_empty_properties(self):
        def my_tool(state: State) -> str:
            return "ok"

        tool = create_tool_from_function(my_tool)
        assert tool.parameters.get("properties", {}) == {}


class TestStateInjectionInToolInvoker:
    def test_state_injected_into_function(self):
        captured = {}

        def my_tool(query: str, state: State) -> str:
            captured["state"] = state
            return f"got {query}"

        tool = create_tool_from_function(my_tool)
        invoker = ToolInvoker(tools=[tool])
        state = State(schema={"x": {"type": str}})

        final_args = invoker._inject_state_args(tool=tool, llm_args={"query": "hello"}, state=state)

        assert final_args["query"] == "hello"
        assert final_args["state"] is state

    def test_state_injection_does_not_affect_other_params(self):
        def my_tool(a: int, b: str, state: State) -> str:
            return f"{a} {b}"

        tool = create_tool_from_function(my_tool)
        invoker = ToolInvoker(tools=[tool])
        state = State(schema={})

        final_args = invoker._inject_state_args(tool=tool, llm_args={"a": 1, "b": "x"}, state=state)

        assert final_args["a"] == 1
        assert final_args["b"] == "x"
        assert final_args["state"] is state

    def test_state_tools_invoked_end_to_end_via_tool_invoker(self):
        ls_tool = LsStateTool()
        write_tool = WriteStateTool()
        read_tool = ReadStateTool()
        invoker = ToolInvoker(tools=[ls_tool, write_tool, read_tool])
        state = State(schema={"answer": {"type": str}}, data={})

        # ls_state
        ls_call = ToolCall(tool_name="ls_state", arguments={})
        result = invoker.run(messages=[ChatMessage.from_assistant(tool_calls=[ls_call])], state=state)
        assert any("answer" in msg.tool_call_results[0].result for msg in result["tool_messages"])

        # write_state
        write_call = ToolCall(tool_name="write_state", arguments={"key": "answer", "value": "42"})
        invoker.run(messages=[ChatMessage.from_assistant(tool_calls=[write_call])], state=state)
        assert state.get("answer") == "42"

        # read_state
        read_call = ToolCall(tool_name="read_state", arguments={"key": "answer"})
        result = invoker.run(messages=[ChatMessage.from_assistant(tool_calls=[read_call])], state=state)
        assert "'42'" in result["tool_messages"][0].tool_call_results[0].result
