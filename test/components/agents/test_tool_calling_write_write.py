# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack.components.agents.state.state import State
from haystack.components.agents.tool_calling import _run_tool, _run_tool_async
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.tools import Tool


def _build_tools():
    def tool_a_fn(dep):
        return {"out": "A-value"}

    def tool_b_fn():
        return {"out": "B-value"}

    def tool_c_fn():
        return {"out": "dep-value"}

    tool_a = Tool(
        name="tool_a",
        description="reads dep, writes shared",
        parameters={"type": "object", "properties": {"dep": {"type": "string"}}, "required": ["dep"]},
        function=tool_a_fn,
        inputs_from_state={"dep": "dep"},
        outputs_to_state={"shared": {"source": "out"}},
    )
    tool_b = Tool(
        name="tool_b",
        description="writes shared, no reads",
        parameters={"type": "object", "properties": {}},
        function=tool_b_fn,
        outputs_to_state={"shared": {"source": "out"}},
    )
    tool_c = Tool(
        name="tool_c",
        description="writes dep",
        parameters={"type": "object", "properties": {}},
        function=tool_c_fn,
        outputs_to_state={"dep": {"source": "out"}},
    )
    return [tool_a, tool_b, tool_c]


def _build_message():
    return ChatMessage.from_assistant(
        tool_calls=[
            ToolCall(tool_name="tool_a", arguments={}),
            ToolCall(tool_name="tool_b", arguments={}),
            ToolCall(tool_name="tool_c", arguments={}),
        ]
    )


def test_write_write_winner_is_last_in_call_order_not_batch_order():
    """tool_a lands in a later batch (it reads dep, written by tool_c), but tool_b — listed after tool_a and
    independent of it — must still win the write-write race on 'shared' (issue #12621)."""
    message = _build_message()
    state = State(schema={"dep": {"type": str}, "shared": {"type": str}})
    _run_tool(messages=[message], state=state, tools=_build_tools(), raise_on_failure=True)
    assert state.get("shared") == "B-value"
    # The read-after-write dependency must still be honored: tool_a read dep after tool_c wrote it.
    assert state.get("dep") == "dep-value"


async def test_write_write_winner_is_last_in_call_order_async():
    message = _build_message()
    state = State(schema={"dep": {"type": str}, "shared": {"type": str}})
    await _run_tool_async(messages=[message], state=state, tools=_build_tools(), raise_on_failure=True)
    assert state.get("shared") == "B-value"
    assert state.get("dep") == "dep-value"
