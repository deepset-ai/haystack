# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated, Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from haystack import component
from haystack.components.agents import Agent
from haystack.components.agents.state import State, replace_values
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.hooks import hook
from haystack.tools import Tool, Toolset, tool


@component
class MockChatGenerator:
    @component.output_types(replies=list[ChatMessage])
    def run(self, messages: list[ChatMessage], tools: list[Tool] | Toolset | None = None, **kwargs) -> dict[str, Any]:
        return {"replies": [ChatMessage.from_assistant("Hello")]}

    @component.output_types(replies=list[ChatMessage])
    async def run_async(
        self, messages: list[ChatMessage], tools: list[Tool] | Toolset | None = None, **kwargs
    ) -> dict[str, Any]:
        return {"replies": [ChatMessage.from_assistant("Hello")]}


@tool
def save(content: Annotated[str, "what to save"]) -> str:
    """Save content."""
    return "saved"


@tool
def final_answer(answer: Annotated[str, "the answer"]) -> str:
    """Provide the final answer."""
    return answer


# --- Module-level hooks (importable so they can be serialized) ---


@hook
def record_a(state: State) -> None:
    state.set("trace", ["a"])


@hook
def record_b(state: State) -> None:
    state.set("trace", ["b"])


@hook
def build_context(state: State) -> None:
    if state.get("step_count") == 0:
        state.set("messages", [ChatMessage.from_system("INJECTED")])


@hook
def record_tool_calls(state: State) -> None:
    # Realistic before_tool use case: audit the tool calls the model is about to run.
    pending = state.data["messages"][-1].tool_calls
    state.set("trace", [tc.tool_name for tc in pending])


@hook
def replace_pending_save_call(state: State) -> None:
    messages = state.data["messages"]
    replacement = ChatMessage.from_assistant(tool_calls=[ToolCall("save", {"content": "changed"})])
    state.set("messages", [*messages[:-1], replacement], handler_override=replace_values)


@hook
def replace_pending_call_with_non_tool_message(state: State) -> None:
    messages = state.data["messages"]
    replacement = ChatMessage.from_assistant("Skipping tool execution.")
    state.set("messages", [*messages[:-1], replacement], handler_override=replace_values)


@hook
def require_save(state: State) -> None:
    if state.get("tool_call_counts", {}).get("save", 0) == 0:
        state.set("messages", [ChatMessage.from_system("You must call save before finishing.")])
        state.set("continue_run", True)


@hook
def always_continue(state: State) -> None:
    state.set("messages", [ChatMessage.from_system("keep going")])
    state.set("continue_run", True)


@hook
def critique(state: State) -> None:
    # Push back on the first final answer to force one more loop.
    if state.get("tool_call_counts", {}).get("final_answer", 0) < 2:
        state.set("messages", [ChatMessage.from_user("Please expand.")])
        state.set("continue_run", True)


class LifecycleHook:
    """A class-based hook that records its lifecycle calls (e.g. opening/closing a client)."""

    def __init__(self) -> None:
        self.warmed = 0
        self.warmed_async = 0
        self.closed = 0
        self.closed_async = 0

    def run(self, state: State) -> None:
        pass

    def warm_up(self) -> None:
        self.warmed += 1

    async def warm_up_async(self) -> None:
        self.warmed_async += 1

    def close(self) -> None:
        self.closed += 1

    async def close_async(self) -> None:
        self.closed_async += 1


def _agent(generator, **kwargs):
    agent = Agent(chat_generator=generator, **kwargs)
    agent.warm_up()
    return agent


class TestAgentHooksValidation:
    def test_invalid_hook_point_raises(self):
        with pytest.raises(ValueError):
            Agent(chat_generator=MockChatGenerator(), hooks={"bogus": [record_a]})

    def test_non_callable_object_raises(self):
        with pytest.raises(TypeError, match="must have a callable 'run"):
            Agent(chat_generator=MockChatGenerator(), hooks={"before_llm": [object()]})

    def test_unwrapped_function_hints_at_hook_decorator(self):
        def my_hook(state: State) -> None:
            pass

        with pytest.raises(TypeError, match="@hook decorator"):
            Agent(chat_generator=MockChatGenerator(), hooks={"before_llm": [my_hook]})

    def test_continue_run_is_a_reserved_state_schema_key(self):
        with pytest.raises(ValueError):
            Agent(chat_generator=MockChatGenerator(), state_schema={"continue_run": {"type": bool}})

    def test_continue_run_is_not_exposed_as_output(self):
        agent = _agent(MockChatGenerator())
        agent.chat_generator.run = MagicMock(return_value={"replies": [ChatMessage.from_assistant("done")]})
        result = agent.run(messages=[ChatMessage.from_user("hi")])
        assert "continue_run" not in result


class TestBeforeLlmHook:
    def test_step_0_only_context_injected_once_across_loops(self):
        agent = _agent(MockChatGenerator(), tools=[save], hooks={"before_llm": [build_context]})
        agent.chat_generator.run = MagicMock(
            side_effect=[
                {"replies": [ChatMessage.from_assistant(tool_calls=[ToolCall("save", {"content": "x"})])]},
                {"replies": [ChatMessage.from_assistant("done")]},
            ]
        )
        result = agent.run(messages=[ChatMessage.from_user("hi")])
        # build_context only injects on step_count == 0, so it appears once despite the two-loop run
        injected = [m for m in result["messages"] if m.is_from("system") and m.text == "INJECTED"]
        assert len(injected) == 1
        assert agent.chat_generator.run.call_count == 2

    def test_hooks_run_in_list_order(self):
        agent = _agent(
            MockChatGenerator(), state_schema={"trace": {"type": list}}, hooks={"before_llm": [record_a, record_b]}
        )
        agent.chat_generator.run = MagicMock(return_value={"replies": [ChatMessage.from_assistant("done")]})
        result = agent.run(messages=[ChatMessage.from_user("hi")])
        assert result["trace"] == ["a", "b"]


class TestBeforeToolHook:
    def test_runs_before_tools_when_model_calls_a_tool(self):
        agent = _agent(
            MockChatGenerator(),
            tools=[save],
            state_schema={"trace": {"type": list}},
            hooks={"before_tool": [record_tool_calls]},
        )
        agent.chat_generator.run = MagicMock(
            side_effect=[
                {"replies": [ChatMessage.from_assistant(tool_calls=[ToolCall("save", {"content": "x"})])]},
                {"replies": [ChatMessage.from_assistant("done")]},
            ]
        )
        result = agent.run(messages=[ChatMessage.from_user("hi")])
        # before_tool sees the pending tool call and does not fire on the final text-only step
        assert result["trace"] == ["save"]
        assert result["tool_call_counts"]["save"] == 1

    def test_rereads_last_state_message_after_hook(self):
        agent = _agent(MockChatGenerator(), tools=[save], hooks={"before_tool": [replace_pending_save_call]})
        agent.chat_generator.run = MagicMock(
            side_effect=[
                {"replies": [ChatMessage.from_assistant(tool_calls=[ToolCall("save", {"content": "original"})])]},
                {"replies": [ChatMessage.from_assistant("done")]},
            ]
        )
        result = agent.run(messages=[ChatMessage.from_user("hi")])
        tool_messages = [m for m in result["messages"] if m.tool_call_result is not None]
        assert len(tool_messages) == 1
        assert tool_messages[0].tool_call_result.origin.arguments == {"content": "changed"}

    def test_skips_tool_execution_when_hook_leaves_last_message_without_tool_calls(self):
        agent = _agent(
            MockChatGenerator(), tools=[save], hooks={"before_tool": [replace_pending_call_with_non_tool_message]}
        )
        agent.chat_generator.run = MagicMock(
            side_effect=[
                {"replies": [ChatMessage.from_assistant(tool_calls=[ToolCall("save", {"content": "x"})])]},
                {"replies": [ChatMessage.from_assistant("done")]},
            ]
        )
        result = agent.run(messages=[ChatMessage.from_user("hi")])
        assert agent.chat_generator.run.call_count == 2
        assert result["tool_call_counts"]["save"] == 0
        assert not [m for m in result["messages"] if m.tool_call_result is not None]
        assert [m.text for m in result["messages"]][-2:] == ["Skipping tool execution.", "done"]


class TestOnExitHook:
    def test_mandatory_tool_forces_extra_step(self):
        agent = _agent(MockChatGenerator(), tools=[save], hooks={"on_exit": [require_save]})
        agent.chat_generator.run = MagicMock(
            side_effect=[
                {"replies": [ChatMessage.from_assistant("Done")]},  # tries to exit without saving
                {"replies": [ChatMessage.from_assistant(tool_calls=[ToolCall("save", {"content": "x"})])]},
                {"replies": [ChatMessage.from_assistant("All finished")]},
            ]
        )
        result = agent.run(messages=[ChatMessage.from_user("hi")])
        assert agent.chat_generator.run.call_count == 3
        assert result["tool_call_counts"]["save"] == 1
        assert result["last_message"].text == "All finished"

    def test_readonly_hook_does_not_change_exit(self):
        fired = []

        def record(state: State) -> None:
            fired.append(1)

        agent = _agent(MockChatGenerator(), tools=[save], hooks={"on_exit": [hook(record)]})
        agent.chat_generator.run = MagicMock(return_value={"replies": [ChatMessage.from_assistant("Final")]})
        agent.run(messages=[ChatMessage.from_user("hi")])
        assert agent.chat_generator.run.call_count == 1
        assert fired == [1]

    def test_critique_on_tool_based_exit(self):
        agent = _agent(
            MockChatGenerator(), tools=[final_answer], exit_conditions=["final_answer"], hooks={"on_exit": [critique]}
        )
        agent.chat_generator.run = MagicMock(
            side_effect=[
                {"replies": [ChatMessage.from_assistant(tool_calls=[ToolCall("final_answer", {"answer": "short"})])]},
                {"replies": [ChatMessage.from_assistant(tool_calls=[ToolCall("final_answer", {"answer": "longer"})])]},
            ]
        )
        result = agent.run(messages=[ChatMessage.from_user("q")])
        # critique forces one extra loop before the final answer is accepted
        assert agent.chat_generator.run.call_count == 2
        assert result["tool_call_counts"]["final_answer"] == 2

    def test_max_agent_steps_bounds_always_cancel_hook(self):
        agent = _agent(MockChatGenerator(), max_agent_steps=3, hooks={"on_exit": [always_continue]})
        agent.chat_generator.run = MagicMock(return_value={"replies": [ChatMessage.from_assistant("text")]})
        agent.run(messages=[ChatMessage.from_user("hi")])
        assert agent.chat_generator.run.call_count == 3


class TestHookReuseAcrossHookPoints:
    def test_same_hook_under_two_hook_points(self):
        counter = []

        def count_call(state: State) -> None:
            counter.append(1)

        counting = hook(count_call)
        agent = _agent(MockChatGenerator(), hooks={"before_llm": [counting], "on_exit": [counting]})
        agent.chat_generator.run = MagicMock(return_value={"replies": [ChatMessage.from_assistant("done")]})
        agent.run(messages=[ChatMessage.from_user("hi")])
        # once for before_llm, once for on_exit (which appends nothing, so the run still exits)
        assert len(counter) == 2


class TestAgentHooksSerde:
    def test_to_dict_from_dict_roundtrip(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
        from haystack.components.generators.chat import OpenAIChatGenerator

        agent = Agent(
            chat_generator=OpenAIChatGenerator(),
            tools=[save],
            hooks={"before_llm": [build_context], "on_exit": [require_save]},
        )
        restored = Agent.from_dict(agent.to_dict())
        assert set(restored.hooks) == {"before_llm", "on_exit"}
        assert restored.hooks["before_llm"][0].function is build_context.function
        assert restored.hooks["on_exit"][0].function is require_save.function


class TestAgentHooksAsync:
    @pytest.mark.asyncio
    async def test_sync_hook_runs_in_async_run(self):
        fired = []

        def record(state: State) -> None:
            fired.append(1)

        agent = _agent(MockChatGenerator(), hooks={"before_llm": [hook(record)]})
        agent.chat_generator.run_async = AsyncMock(return_value={"replies": [ChatMessage.from_assistant("done")]})
        await agent.run_async(messages=[ChatMessage.from_user("hi")])
        assert fired == [1]

    @pytest.mark.asyncio
    async def test_async_on_exit_hook_forces_extra_step(self):
        async def require_save_async(state: State) -> None:
            if state.get("tool_call_counts", {}).get("save", 0) == 0:
                state.set("messages", [ChatMessage.from_system("call save first")])
                state.set("continue_run", True)

        agent = _agent(MockChatGenerator(), tools=[save], hooks={"on_exit": [hook(require_save_async)]})
        agent.chat_generator.run_async = AsyncMock(
            side_effect=[
                {"replies": [ChatMessage.from_assistant("Done")]},
                {"replies": [ChatMessage.from_assistant(tool_calls=[ToolCall("save", {"content": "x"})])]},
                {"replies": [ChatMessage.from_assistant("All finished")]},
            ]
        )
        result = await agent.run_async(messages=[ChatMessage.from_user("hi")])
        assert agent.chat_generator.run_async.call_count == 3
        assert result["tool_call_counts"]["save"] == 1


class TestAgentHookLifecycle:
    def test_init_does_not_warm_up(self):
        h = LifecycleHook()
        Agent(chat_generator=MockChatGenerator(), hooks={"before_llm": [h]})
        assert h.warmed == 0

    def test_warm_up_warms_hooks_once(self):
        h = LifecycleHook()
        agent = Agent(chat_generator=MockChatGenerator(), hooks={"before_llm": [h]})
        agent.warm_up()
        agent.warm_up()
        assert h.warmed == 1

    def test_reused_hook_warmed_once(self):
        h = LifecycleHook()
        agent = Agent(chat_generator=MockChatGenerator(), hooks={"before_llm": [h], "on_exit": [h]})
        agent.warm_up()
        assert h.warmed == 1

    def test_close_closes_hooks(self):
        h = LifecycleHook()
        agent = Agent(chat_generator=MockChatGenerator(), hooks={"before_llm": [h]})
        agent.close()
        assert h.closed == 1


class TestAgentHookLifecycleAsync:
    @pytest.mark.asyncio
    async def test_warm_up_async_prefers_async(self):
        h = LifecycleHook()
        agent = Agent(chat_generator=MockChatGenerator(), hooks={"before_llm": [h]})
        await agent.warm_up_async()
        assert h.warmed_async == 1
        assert h.warmed == 0

    @pytest.mark.asyncio
    async def test_close_async_prefers_async(self):
        h = LifecycleHook()
        agent = Agent(chat_generator=MockChatGenerator(), hooks={"before_llm": [h]})
        await agent.close_async()
        assert h.closed_async == 1
        assert h.closed == 0
