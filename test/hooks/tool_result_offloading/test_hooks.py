# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import base64
from pathlib import Path
from typing import Annotated
from unittest.mock import AsyncMock, MagicMock

import pytest

from haystack.components.agents import Agent
from haystack.components.agents.state.state import State
from haystack.components.generators.chat import MockChatGenerator
from haystack.dataclasses import ChatMessage, FileContent, ImageContent, TextContent, ToolCall
from haystack.hooks.tool_result_offloading import (
    RESULT_STORE_CONTEXT_KEY,
    AlwaysOffload,
    FileSystemToolResultStore,
    NeverOffload,
    OffloadOverChars,
    ToolResultOffloadHook,
    ToolResultStore,
)
from haystack.tools import tool


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


# A 1x1 PNG and a minimal PDF.
PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
PDF_BYTES = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\n\xde\xad\xbe\xef\n%%EOF\n"


class TextOnlyToolResultStore(ToolResultStore):
    """A store that only holds text, leaving `supports_binary_content` at its False default."""

    def __init__(self) -> None:
        self.data: dict[str, str] = {}

    def write(self, *, key: str, content: str) -> str:
        self.data[key] = content
        return key

    def read(self, reference: str) -> str:
        return self.data[reference]


def _image_block() -> ImageContent:
    return ImageContent(base64_image=base64.b64encode(PNG_BYTES).decode("utf-8"), mime_type="image/png")


def _file_block(filename: str | None = "report.pdf") -> FileContent:
    return FileContent(
        base64_data=base64.b64encode(PDF_BYTES).decode("utf-8"), mime_type="application/pdf", filename=filename
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

    def test_mixed_result_offloads_every_block_to_its_own_entry(self, tmp_path):
        store = FileSystemToolResultStore(root=tmp_path)
        hook = ToolResultOffloadHook(store=store, offload_strategies={"*": AlwaysOffload()}, preview_chars=4)
        content = [TextContent("caption"), _image_block(), _file_block()]
        message = ChatMessage.from_tool(tool_result=content, origin=ToolCall(tool_name="a", arguments={}, id="1"))
        state = _state_with_messages([message])
        hook.run(state)
        offloaded = state.data["messages"][0]
        references = offloaded.meta["tool_result_offloaded"]
        assert len(references) == 3
        assert [Path(reference).name for reference in references] == ["1_a_1_0.txt", "1_a_1_1.png", "1_a_1_2.pdf"]
        assert store.read(references[0]) == "caption"
        assert store.read(references[1]) == PNG_BYTES
        assert store.read(references[2]) == PDF_BYTES
        pointer = offloaded.tool_call_result.result
        assert pointer.startswith("Tool result offloaded to 3 files:")
        assert f"1. text (7 characters) at '{references[0]}'. Preview: capt..." in pointer
        assert f"2. image/png ({len(PNG_BYTES)} bytes) at '{references[1]}'" in pointer
        assert f"3. application/pdf named 'report.pdf' ({len(PDF_BYTES)} bytes) at '{references[2]}'" in pointer

    @pytest.mark.parametrize(
        "block, expected_extension",
        [
            pytest.param(FileContent(base64_data="aGk=", mime_type="text/csv", filename=None), ".csv", id="mime_type"),
            pytest.param(
                FileContent(base64_data="aGk=", mime_type="application/pdf", filename="notes.md"), ".md", id="filename"
            ),
            pytest.param(FileContent(base64_data="aGk=", mime_type=None, filename=None), ".bin", id="fallback"),
        ],
    )
    def test_binary_block_extension(self, tmp_path, block, expected_extension):
        hook = ToolResultOffloadHook(
            store=FileSystemToolResultStore(root=tmp_path), offload_strategies={"*": AlwaysOffload()}
        )
        message = ChatMessage.from_tool(tool_result=[block], origin=ToolCall(tool_name="a", arguments={}, id="1"))
        state = _state_with_messages([message])
        hook.run(state)
        reference = state.data["messages"][0].meta["tool_result_offloaded"][0]
        assert Path(reference).suffix == expected_extension

    def test_policy_sizes_result_by_its_base64_payload(self, tmp_path):
        # The base64 payload is what occupies the context window, so a threshold below it must trigger an offload
        # while one above it must not.
        image = _image_block()
        payload_length = len(image.base64_image)
        message = ChatMessage.from_tool(tool_result=[image], origin=ToolCall(tool_name="a", arguments={}, id="1"))
        offloading_hook = ToolResultOffloadHook(
            store=FileSystemToolResultStore(root=tmp_path / "over"),
            offload_strategies={"*": OffloadOverChars(payload_length - 1)},
        )
        keeping_hook = ToolResultOffloadHook(
            store=FileSystemToolResultStore(root=tmp_path / "under"),
            offload_strategies={"*": OffloadOverChars(payload_length)},
        )
        over_state = _state_with_messages([message])
        offloading_hook.run(over_state)
        under_state = _state_with_messages([message])
        keeping_hook.run(under_state)
        assert over_state.data["messages"][0].tool_call_result.result.startswith("Tool result offloaded")
        assert under_state.data["messages"][0].tool_call_result.result == [image]

    @pytest.mark.parametrize(
        "empty_result",
        ["", [], [TextContent(text="")], [TextContent(text=""), TextContent(text="")]],
        ids=["empty_string", "empty_list", "single_empty_text_block", "several_empty_text_blocks"],
    )
    def test_empty_result_is_not_offloaded(self, tmp_path, empty_result):
        hook = ToolResultOffloadHook(
            store=FileSystemToolResultStore(root=tmp_path), offload_strategies={"*": AlwaysOffload()}
        )
        message = ChatMessage.from_tool(tool_result=empty_result, origin=ToolCall(tool_name="a", arguments={}, id="1"))
        state = _state_with_messages([message])
        hook.run(state)
        assert state.data["messages"][0].tool_call_result.result == empty_result
        assert not list(Path(tmp_path).iterdir())

    def test_empty_text_alongside_image_is_offloaded(self, tmp_path):
        hook = ToolResultOffloadHook(
            store=FileSystemToolResultStore(root=tmp_path), offload_strategies={"*": AlwaysOffload()}
        )
        image = ImageContent(base64_image=base64.b64encode(b"PNGDATA").decode(), mime_type="image/png")
        message = ChatMessage.from_tool(
            tool_result=[TextContent(text=""), image], origin=ToolCall(tool_name="a", arguments={}, id="1")
        )
        state = _state_with_messages([message])
        hook.run(state)
        assert state.data["messages"][0].tool_call_result.result.startswith("Tool result offloaded")

    def test_id_less_parallel_calls_do_not_collide(self, tmp_path):
        store = FileSystemToolResultStore(root=tmp_path)
        hook = ToolResultOffloadHook(store=store, offload_strategies={"*": AlwaysOffload()})
        first = ChatMessage.from_tool(tool_result="FIRST" * 20, origin=ToolCall(tool_name="a", arguments={}, id=None))
        second = ChatMessage.from_tool(tool_result="SECOND" * 20, origin=ToolCall(tool_name="a", arguments={}, id=None))
        state = _state_with_messages([first, second])
        hook.run(state)
        refs = [m.meta["tool_result_offloaded"][0] for m in state.data["messages"]]
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

    def test_second_offload_hook_does_not_reoffload_pointer(self, tmp_path):
        # Two offload hooks under `after_tool` run in sequence on the same state. The first offloads the result and
        # marks it; the `_OFFLOADED_META_KEY` marker stops the second from offloading the pointer text again.
        store = FileSystemToolResultStore(root=tmp_path)
        first_hook = ToolResultOffloadHook(store=store, offload_strategies={"*": AlwaysOffload()})
        second_hook = ToolResultOffloadHook(store=store, offload_strategies={"*": AlwaysOffload()})
        state = _state_with_messages([_tool_message("a", "A" * 50)])
        first_hook.run(state)
        pointer = state.data["messages"][0].tool_call_result.result
        second_hook.run(state)
        assert state.data["messages"][0].tool_call_result.result == pointer
        assert len(list(Path(tmp_path).iterdir())) == 1

    def test_pointer_contains_reference_size_and_preview(self, tmp_path):
        hook = ToolResultOffloadHook(
            store=FileSystemToolResultStore(root=tmp_path), offload_strategies={"*": AlwaysOffload()}, preview_chars=5
        )
        state = _state_with_messages([_tool_message("a", "ABCDEFGH")])
        hook.run(state)
        message = state.data["messages"][0]
        reference = message.meta["tool_result_offloaded"][0]
        pointer = message.tool_call_result.result
        assert pointer == f"Tool result offloaded to text (8 characters) at '{reference}'. Preview: ABCDE..."

    def test_concurrent_runs_are_isolated_by_per_run_store(self, tmp_path):
        # One shared hook instance, two runs each supplying its own store via hook_context. Even with identical store
        # keys (same tool, id and step), each run writes to and reads from its own store — no cross-run collision.
        hook = ToolResultOffloadHook(
            store=FileSystemToolResultStore(root=tmp_path / "shared"), offload_strategies={"*": AlwaysOffload()}
        )
        store_a = FileSystemToolResultStore(root=tmp_path / "a")
        store_b = FileSystemToolResultStore(root=tmp_path / "b")
        state_a = _state_with_messages([_tool_message("t", "AAA" * 20, call_id="x")])
        state_a.data["hook_context"] = {RESULT_STORE_CONTEXT_KEY: store_a}
        state_b = _state_with_messages([_tool_message("t", "BBB" * 20, call_id="x")])
        state_b.data["hook_context"] = {RESULT_STORE_CONTEXT_KEY: store_b}
        hook.run(state_a)
        hook.run(state_b)
        ref_a = state_a.data["messages"][0].meta["tool_result_offloaded"][0]
        ref_b = state_b.data["messages"][0].meta["tool_result_offloaded"][0]
        assert store_a.read(ref_a) == "AAA" * 20
        assert store_b.read(ref_b) == "BBB" * 20
        assert not (tmp_path / "shared").exists()

    def test_hook_context_store_overrides_constructor_store(self, tmp_path):
        default_store = FileSystemToolResultStore(root=tmp_path / "default")
        request_store = FileSystemToolResultStore(root=tmp_path / "request")
        hook = ToolResultOffloadHook(store=default_store, offload_strategies={"*": AlwaysOffload()})
        state = _state_with_messages([_tool_message("a", "A" * 50)])
        state.data["hook_context"] = {RESULT_STORE_CONTEXT_KEY: request_store}
        hook.run(state)
        assert (tmp_path / "request").exists()
        assert not (tmp_path / "default").exists()


class TestToolResultOffloadHookWithTextOnlyStore:
    def test_text_results_are_offloaded(self, caplog):
        store = TextOnlyToolResultStore()
        hook = ToolResultOffloadHook(store=store, offload_strategies={"*": AlwaysOffload()})
        state = _state_with_messages([_tool_message("a", "A" * 50)])
        hook.run(state)
        offloaded = state.data["messages"][0]
        assert offloaded.tool_call_result.result.startswith("Tool result offloaded")
        assert store.read(offloaded.meta["tool_result_offloaded"][0]) == "A" * 50
        assert not caplog.records

    def test_image_and_file_results_stay_in_context(self, caplog):
        hook = ToolResultOffloadHook(store=TextOnlyToolResultStore(), offload_strategies={"*": AlwaysOffload()})
        content = [TextContent("caption"), _file_block()]
        message = ChatMessage.from_tool(tool_result=content, origin=ToolCall(tool_name="a", arguments={}, id="1"))
        state = _state_with_messages([message])
        hook.run(state)
        assert state.data["messages"][0].tool_call_result.result == content
        assert "does not support binary content" in caplog.text


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
        agent = Agent(chat_generator=MockChatGenerator("done"), tools=[big_tool], hooks={"after_tool": [hook]})
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
        agent = Agent(chat_generator=MockChatGenerator("done"), tools=[big_tool], hooks={"after_tool": [hook]})
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
