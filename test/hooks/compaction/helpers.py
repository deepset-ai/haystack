# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack.components.agents.state.state import State
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.hooks.compaction.utils import _COMPACTION_META_KEY
from haystack.token_counters.utils import _rendered_conversation, _rendered_tools
from haystack.tools import ToolsType

# The keys of the Agent's State that compaction reads or writes.
_SCHEMA = {
    "messages": {"type": list[ChatMessage]},
    "step_count": {"type": int},
    "context_tokens": {"type": int},
    "token_usage": {"type": dict},
    "tool_call_counts": {"type": dict},
}


class FakeCounter:
    """A counter with a fixed, obvious rate, so tests can assert exact numbers without depending on a tokenizer."""

    def __init__(self, *, chars_per_token: int = 4) -> None:
        self.chars_per_token = chars_per_token
        self.calls = 0

    def count(self, messages: list[ChatMessage], tools: ToolsType | None = None) -> int:
        self.calls += 1
        if not messages and not tools:
            return 0
        return len(_rendered_conversation(messages) + _rendered_tools(tools)) // self.chars_per_token

    def to_dict(self) -> dict[str, Any]:
        return {"type": "test.hooks.compaction.helpers.FakeCounter", "init_parameters": {}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FakeCounter":
        return cls()


def tool_call(*call_ids: str, name: str = "search", arguments: dict[str, Any] | None = None) -> ChatMessage:
    """An assistant message requesting one tool call per given id, so several ids give a parallel batch."""
    return ChatMessage.from_assistant(
        tool_calls=[ToolCall(tool_name=name, arguments=arguments or {}, id=call_id) for call_id in call_ids]
    )


def tool_result(result: str, *, call_id: str = "c1", name: str = "search", error: bool = False) -> ChatMessage:
    """A tool-result message answering the call with the given id."""
    return ChatMessage.from_tool(
        tool_result=result, origin=ToolCall(tool_name=name, arguments={}, id=call_id), error=error
    )


def make_state(messages: list[ChatMessage], **data: Any) -> State:
    """A State shaped like the Agent's, holding `messages` and whatever run metadata a test overrides."""
    base = {"messages": messages, "step_count": 2, "context_tokens": 0, "token_usage": {}, "tool_call_counts": {}}
    return State(schema=_SCHEMA, data={**base, **data})


def long_conversation() -> list[ChatMessage]:
    """Six messages: a system prefix, a user turn, then two tool round-trips."""
    return [
        ChatMessage.from_system("rules"),
        ChatMessage.from_user("start"),
        tool_call("c1"),
        tool_result("first result", call_id="c1"),
        tool_call("c2"),
        tool_result("second result", call_id="c2"),
    ]


def count_markers(messages: list[ChatMessage]) -> int:
    """How many messages carry a compaction marker."""
    return sum(_COMPACTION_META_KEY in message.meta for message in messages)
