# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Annotated, Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from haystack import component
from haystack.components.agents import Agent
from haystack.components.agents.state.state import State
from haystack.dataclasses import ChatMessage, ImageContent, TextContent, ToolCall
from haystack.hooks.tool_result_offloading import (
    RESULT_STORE_CONTEXT_KEY,
    AlwaysOffload,
    FileSystemToolResultStore,
    NeverOffload,
    OffloadOverChars,
    ToolResultOffloadHook,
)
from haystack.tools import Tool, Toolset, tool


@component
class MockChatGenerator:
    @component.output_types(replies=list[ChatMessage])
    def run(self, messages: list[ChatMessage], tools: list[Tool] | Toolset | None = None, **kwargs) -> dict[str, Any]:
        return {"replies": [ChatMessage.from_assistant("done")]}

    @component.output_types(replies=list[ChatMessage])
    async def run_async(
        self, messages: list[ChatMessage], tools: list[Tool] | Toolset | None = None, **kwargs
    ) -> dict[str, Any]:
        return {"replies": [ChatMessage.from_assistant("done")]}


@tool
def big_tool(query: Annotated[str, "the query"]) -> str:
    """Return a large result."""
    return "R" * 500


def _state_with_messages(messages: list[ChatMessage]) -> State:
    return State(schema={"messages": {"type": list[ChatMessage]}}, data={"messages": messages, "step_count": 1})


def _tool_message(tool_name: str, result: str, *, error: bool = False, call_id: str = "c1") -> ChatMessage:
    return ChatMessage.from_tool(
        tool_result=result, origin=ToolCall(tool_name=tool_name, arguments={}, id=call_id), error=error
    )


class TestToolResultOffloadHookRouting:
    def test_exact_tuple_and_wildcard_keys(self, tmp_path):
        hook = ToolResultOffloadHook(
            store=FileSystemToolResultStore(root=tmp_path),
            offload_strategies={
                "a": AlwaysOffload(),
                ("b", "c"): AlwaysOffload(),
                "d": NeverOffload(),
                "*": AlwaysOffload(),
            },
        )
        state = _state_with_messages(
            [
                _tool_message("a", "A" * 50, call_id="1"),  # exact -> offload
                _tool_message("b", "B" * 50, call_id="2"),  # tuple -> offload
                _tool_message("d", "D" * 50, call_id="3"),  # exact NeverOffload -> keep
                _tool_message("z", "Z" * 50, call_id="4"),  # wildcard -> offload
            ]
        )
        hook.run(state)
        results = [m.tool_call_result.result for m in state.data["messages"]]
        assert results[0].startswith("Tool result offloaded")
        assert results[1].startswith("Tool result offloaded")
        assert results[2] == "D" * 50
        assert results[3].startswith("Tool result offloaded")

    def test_tool_without_matching_key_is_not_offloaded(self, tmp_path):
        hook = ToolResultOffloadHook(
            store=FileSystemToolResultStore(root=tmp_path), offload_strategies={"a": AlwaysOffload()}
        )
        state = _state_with_messages([_tool_message("b", "B" * 50)])
        hook.run(state)
        assert state.data["messages"][0].tool_call_result.result == "B" * 50

    def test_over_chars_threshold(self, tmp_path):
        hook = ToolResultOffloadHook(
            store=FileSystemToolResultStore(root=tmp_path), offload_strategies={"*": OffloadOverChars(10)}
        )
        state = _state_with_messages(
            [_tool_message("a", "x" * 10, call_id="1"), _tool_message("a", "x" * 11, call_id="2")]
        )
        hook.run(state)
        results = [m.tool_call_result.result for m in state.data["messages"]]
        assert results[0] == "x" * 10
        assert results[1].startswith("Tool result offloaded")


class TestToolResultOffloadHookBehavior:
    def test_error_results_are_never_offloaded(self, tmp_path):
        hook = ToolResultOffloadHook(
            store=FileSystemToolResultStore(root=tmp_path), offload_strategies={"*": AlwaysOffload()}
        )
        state = _state_with_messages([_tool_message("a", "boom", error=True)])
        hook.run(state)
        assert state.data["messages"][0].tool_call_result.result == "boom"

    def test_offloads_sequence_of_text_content(self, tmp_path):
        store = FileSystemToolResultStore(root=tmp_path)
        hook = ToolResultOffloadHook(store=store, offload_strategies={"*": AlwaysOffload()})
        message = ChatMessage.from_tool(
            tool_result=[TextContent("A" * 30), TextContent("B" * 30)],
            origin=ToolCall(tool_name="a", arguments={}, id="1"),
        )
        state = _state_with_messages([message])
        hook.run(state)
        offloaded = state.data["messages"][0]
        assert offloaded.tool_call_result.result.startswith("Tool result offloaded")
        assert "60 characters" in offloaded.tool_call_result.result
        assert store.read(offloaded.meta["tool_result_offloaded"]) == "A" * 30 + "B" * 30

    def test_result_with_image_content_is_not_offloaded(self, tmp_path):
        hook = ToolResultOffloadHook(
            store=FileSystemToolResultStore(root=tmp_path), offload_strategies={"*": AlwaysOffload()}
        )
        content = [TextContent("caption"), ImageContent(base64_image="aGVsbG8=", mime_type="image/png")]
        message = ChatMessage.from_tool(tool_result=content, origin=ToolCall(tool_name="a", arguments={}, id="1"))
        state = _state_with_messages([message])
        hook.run(state)
        assert state.data["messages"][0].tool_call_result.result == content
        assert not list(Path(tmp_path).iterdir())

    def test_id_less_parallel_calls_do_not_collide(self, tmp_path):
        store = FileSystemToolResultStore(root=tmp_path)
        hook = ToolResultOffloadHook(store=store, offload_strategies={"*": AlwaysOffload()})
        first = ChatMessage.from_tool(tool_result="FIRST" * 20, origin=ToolCall(tool_name="a", arguments={}, id=None))
        second = ChatMessage.from_tool(tool_result="SECOND" * 20, origin=ToolCall(tool_name="a", arguments={}, id=None))
        state = _state_with_messages([first, second])
        hook.run(state)
        refs = [m.meta["tool_result_offloaded"] for m in state.data["messages"]]
        assert refs[0] != refs[1]
        assert store.read(refs[0]) == "FIRST" * 20
        assert store.read(refs[1]) == "SECOND" * 20
        assert len(list(Path(tmp_path).iterdir())) == 2

    def test_only_trailing_tool_results_are_offloaded(self, tmp_path):
        store = FileSystemToolResultStore(root=tmp_path)
        hook = ToolResultOffloadHook(store=store, offload_strategies={"*": AlwaysOffload()})
        # A tool result from a prior turn, then an assistant message, then this step's fresh tool result.
        history = _tool_message("old", "H" * 50, call_id="old1")
        assistant = ChatMessage.from_assistant(tool_calls=[ToolCall("a", {}, id="c1")])
        fresh = _tool_message("a", "F" * 50, call_id="c1")
        state = _state_with_messages([history, assistant, fresh])
        hook.run(state)
        out = state.data["messages"]
        assert out[0].tool_call_result.result == "H" * 50
        assert out[2].tool_call_result.result.startswith("Tool result offloaded")
        assert len(list(Path(tmp_path).iterdir())) == 1

    def test_offloading_is_idempotent_across_runs(self, tmp_path):
        hook = ToolResultOffloadHook(
            store=FileSystemToolResultStore(root=tmp_path), offload_strategies={"*": AlwaysOffload()}
        )
        state = _state_with_messages([_tool_message("a", "A" * 50)])
        hook.run(state)
        first = state.data["messages"][0].tool_call_result.result
        hook.run(state)
        assert state.data["messages"][0].tool_call_result.result == first
        assert len(list(Path(tmp_path).iterdir())) == 1

    def test_pointer_contains_reference_size_and_preview(self, tmp_path):
        hook = ToolResultOffloadHook(
            store=FileSystemToolResultStore(root=tmp_path), offload_strategies={"*": AlwaysOffload()}, preview_chars=5
        )
        state = _state_with_messages([_tool_message("a", "ABCDEFGH")])
        hook.run(state)
        message = state.data["messages"][0]
        reference = message.meta["tool_result_offloaded"]
        pointer = message.tool_call_result.result
        assert reference in pointer
        assert "8 characters" in pointer
        assert "ABCDE..." in pointer

    def test_hook_context_store_overrides_constructor_store(self, tmp_path):
        default_store = FileSystemToolResultStore(root=tmp_path / "default")
        request_store = FileSystemToolResultStore(root=tmp_path / "request")
        hook = ToolResultOffloadHook(store=default_store, offload_strategies={"*": AlwaysOffload()})
        state = _state_with_messages([_tool_message("a", "A" * 50)])
        state.data["hook_context"] = {RESULT_STORE_CONTEXT_KEY: request_store}
        hook.run(state)
        assert (tmp_path / "request").exists()
        assert not (tmp_path / "default").exists()


class TestToolResultOffloadHookSerde:
    def test_to_dict_from_dict_roundtrip(self, tmp_path):
        hook = ToolResultOffloadHook(
            store=FileSystemToolResultStore(root=tmp_path),
            offload_strategies={"a": AlwaysOffload(), ("b", "c"): OffloadOverChars(100), "*": NeverOffload()},
            preview_chars=42,
        )
        restored = ToolResultOffloadHook.from_dict(hook.to_dict())
        assert restored.preview_chars == 42
        assert isinstance(restored.store, FileSystemToolResultStore)
        assert set(restored.offload_strategies) == {"a", ("b", "c"), "*"}
        assert isinstance(restored.offload_strategies[("b", "c")], OffloadOverChars)
        assert restored.offload_strategies[("b", "c")].threshold == 100


class TestToolResultOffloadHookInAgent:
    def test_offloads_tool_result_seen_by_next_llm_call(self, tmp_path):
        hook = ToolResultOffloadHook(
            store=FileSystemToolResultStore(root=tmp_path), offload_strategies={"*": AlwaysOffload()}
        )
        agent = Agent(chat_generator=MockChatGenerator(), tools=[big_tool], hooks={"after_tool": [hook]})
        agent.warm_up()
        agent.chat_generator.run = MagicMock(
            side_effect=[
                {"replies": [ChatMessage.from_assistant(tool_calls=[ToolCall("big_tool", {"query": "x"})])]},
                {"replies": [ChatMessage.from_assistant("done")]},
            ]
        )
        agent.run(messages=[ChatMessage.from_user("hi")])
        second_call_messages = agent.chat_generator.run.call_args_list[1].kwargs["messages"]
        offloaded = [m for m in second_call_messages if m.tool_call_result is not None]
        assert offloaded[0].tool_call_result.result.startswith("Tool result offloaded")
        assert len(list(Path(tmp_path).iterdir())) == 1


class TestToolResultOffloadHookInAgentAsync:
    @pytest.mark.asyncio
    async def test_offloads_tool_result_async(self, tmp_path):
        hook = ToolResultOffloadHook(
            store=FileSystemToolResultStore(root=tmp_path), offload_strategies={"*": AlwaysOffload()}
        )
        agent = Agent(chat_generator=MockChatGenerator(), tools=[big_tool], hooks={"after_tool": [hook]})
        agent.warm_up()
        agent.chat_generator.run_async = AsyncMock(
            side_effect=[
                {"replies": [ChatMessage.from_assistant(tool_calls=[ToolCall("big_tool", {"query": "x"})])]},
                {"replies": [ChatMessage.from_assistant("done")]},
            ]
        )
        result = await agent.run_async(messages=[ChatMessage.from_user("hi")])
        offloaded = [m for m in result["messages"] if m.tool_call_result is not None]
        assert offloaded[0].tool_call_result.result.startswith("Tool result offloaded")
