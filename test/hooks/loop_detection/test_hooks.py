# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from typing import Annotated, Any
from unittest.mock import MagicMock

import pytest

from haystack.components.agents import Agent
from haystack.components.agents.state import State
from haystack.components.generators.chat import MockChatGenerator
from haystack.dataclasses import ChatMessage, ChatRole, ToolCall
from haystack.hooks.loop_detection import LoopDetectionResult, ToolLoopDetectionHook
from haystack.hooks.loop_detection.detector import detect_tool_loop
from haystack.hooks.loop_detection.hooks import _EXIT_REASON_TOOL_LOOP, _LOOP_WARNING_META_KEY
from haystack.hooks.utils import _SYSTEM_REMINDER_META_KEY
from haystack.tools import tool

pytestmark = pytest.mark.filterwarnings("ignore::haystack.utils.experimental.ExperimentalWarning")


@tool
def search(query: Annotated[str, "search query"]) -> str:
    """Search for information."""
    return f"results for {query}"


@tool
def fetch(url: Annotated[str, "url to fetch"]) -> str:
    """Fetch a webpage."""
    return f"content of {url}"


@tool
def flaky(attempt: Annotated[int, "attempt number"]) -> str:
    """Always raise an error to simulate a failing tool."""
    raise RuntimeError(f"upstream failed on attempt {attempt}")


def _tool_step_messages(tool_calls: list[ToolCall], *, error_tools: set[str] | None = None) -> list[ChatMessage]:
    error_tools = error_tools or set()
    msgs = [ChatMessage.from_assistant(tool_calls=tool_calls)]
    for tc in tool_calls:
        is_error = tc.tool_name in error_tools
        msgs.append(ChatMessage.from_tool(tool_result="boom" if is_error else "ok", origin=tc, error=is_error))
    return msgs


def _state_with_messages(messages: list[ChatMessage]) -> State:
    schema = {"messages": {"type": list[ChatMessage]}, "stop_run": {"type": str}}
    return State(schema=schema, data={"messages": list(messages)})


class TestDetectToolLoop:
    def test_exact_repeat_ignores_dict_key_and_parallel_call_order(self):
        tc_a1 = ToolCall(tool_name="search", arguments={"a": 1, "b": 2}, id="1")
        tc_b1 = ToolCall(tool_name="fetch", arguments={"url": "https://example.com"}, id="2")
        tc_b2 = ToolCall(tool_name="fetch", arguments={"url": "https://example.com"}, id="3")
        tc_a2 = ToolCall(tool_name="search", arguments={"b": 2, "a": 1}, id="4")

        messages = [
            ChatMessage.from_user("find info"),
            *_tool_step_messages([tc_a1, tc_b1]),
            *_tool_step_messages([tc_b2, tc_a2]),
        ]
        result, action = detect_tool_loop(
            messages=messages,
            warn_threshold=2,
            stop_threshold=3,
            max_cycle_length=3,
            window_size=20,
            detect_consecutive_errors=True,
        )
        assert action == "warn"
        assert isinstance(result, LoopDetectionResult)
        assert result.kind == "exact_repeat"
        assert result.repetitions == 2
        assert result.cycle_length == 1
        assert result.tool_names == ("fetch", "search")

    def test_ping_pong_cycle_detected_and_disableable(self):
        a = ToolCall(tool_name="search", arguments={"query": "a"})
        b = ToolCall(tool_name="fetch", arguments={"url": "https://a.test"})
        messages = [
            ChatMessage.from_user("loop"),
            *_tool_step_messages([a]),
            *_tool_step_messages([b]),
            *_tool_step_messages([a]),
            *_tool_step_messages([b]),
        ]
        result, action = detect_tool_loop(
            messages=messages,
            warn_threshold=2,
            stop_threshold=3,
            max_cycle_length=3,
            window_size=20,
            detect_consecutive_errors=True,
        )
        assert action == "warn"
        assert result is not None
        assert result.kind == "ping_pong"
        assert result.cycle_length == 2
        assert result.repetitions == 2
        assert result.tool_names == ("search", "fetch")

        no_result, no_action = detect_tool_loop(
            messages=messages,
            warn_threshold=2,
            stop_threshold=3,
            max_cycle_length=1,
            window_size=20,
            detect_consecutive_errors=True,
        )
        assert no_result is None
        assert no_action is None

    def test_consecutive_errors_detected_even_with_varying_arguments(self):
        messages = [
            ChatMessage.from_user("run flaky"),
            *_tool_step_messages([ToolCall(tool_name="flaky", arguments={"attempt": 1})], error_tools={"flaky"}),
            *_tool_step_messages([ToolCall(tool_name="flaky", arguments={"attempt": 2})], error_tools={"flaky"}),
            *_tool_step_messages([ToolCall(tool_name="flaky", arguments={"attempt": 3})], error_tools={"flaky"}),
        ]
        result, action = detect_tool_loop(
            messages=messages,
            warn_threshold=2,
            stop_threshold=3,
            max_cycle_length=3,
            window_size=20,
            detect_consecutive_errors=True,
        )
        assert action == "stop"
        assert result is not None
        assert result.kind == "consecutive_error"
        assert result.tool_names == ("flaky",)
        assert result.repetitions == 3

        disabled_result, disabled_action = detect_tool_loop(
            messages=messages,
            warn_threshold=2,
            stop_threshold=3,
            max_cycle_length=3,
            window_size=20,
            detect_consecutive_errors=False,
        )
        assert disabled_result is None
        assert disabled_action is None

    def test_new_user_turn_resets_window_but_system_reminder_does_not(self):
        tc = ToolCall(tool_name="search", arguments={"query": "same"})
        multi_turn_messages = [
            ChatMessage.from_user("turn 1"),
            *_tool_step_messages([tc]),
            ChatMessage.from_assistant("turn 1 answer"),
            ChatMessage.from_user("turn 2"),
            *_tool_step_messages([tc]),
        ]
        result, action = detect_tool_loop(
            messages=multi_turn_messages,
            warn_threshold=2,
            stop_threshold=3,
            max_cycle_length=3,
            window_size=20,
            detect_consecutive_errors=True,
        )
        assert result is None
        assert action is None

        reminder = ChatMessage.from_user(
            text="<system-reminder>\nkeep going\n</system-reminder>", meta={_SYSTEM_REMINDER_META_KEY: True}
        )
        same_turn_with_reminder = [
            ChatMessage.from_user("turn 1"),
            *_tool_step_messages([tc]),
            reminder,
            *_tool_step_messages([tc]),
        ]
        result2, action2 = detect_tool_loop(
            messages=same_turn_with_reminder,
            warn_threshold=2,
            stop_threshold=3,
            max_cycle_length=3,
            window_size=20,
            detect_consecutive_errors=True,
        )
        assert action2 == "warn"
        assert result2 is not None
        assert result2.kind == "exact_repeat"
        assert result2.repetitions == 2

    def test_handles_non_json_serializable_tool_arguments(self):
        tc1 = ToolCall(tool_name="search", arguments={"payload": {1, 2, 3}})
        tc2 = ToolCall(tool_name="search", arguments={"payload": {1, 2, 3}})
        messages = [ChatMessage.from_user("hi"), *_tool_step_messages([tc1]), *_tool_step_messages([tc2])]
        result, action = detect_tool_loop(
            messages=messages,
            warn_threshold=2,
            stop_threshold=3,
            max_cycle_length=3,
            window_size=20,
            detect_consecutive_errors=True,
        )
        assert action == "warn"
        assert result is not None
        assert result.kind == "exact_repeat"


class TestToolLoopDetectionHook:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"stop_threshold": 1}, "`stop_threshold` must be at least 2"),
            ({"warn_threshold": 1, "stop_threshold": 3}, "`warn_threshold` must be at least 2 or None"),
            (
                {"warn_threshold": 3, "stop_threshold": 3},
                r"`warn_threshold` \(3\) must be less than `stop_threshold` \(3\)\.",
            ),
            ({"max_cycle_length": 0}, "`max_cycle_length` must be at least 1"),
            ({"stop_threshold": 4, "window_size": 3}, r"`window_size` \(3\) must be greater than or equal"),
        ],
    )
    def test_invalid_init_parameters_raise(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            ToolLoopDetectionHook(**kwargs)

    def test_invalid_hook_point_rejected_by_agent(self):
        with pytest.raises(ValueError, match="only supports: before_llm"):
            Agent(chat_generator=MockChatGenerator(), tools=[search], hooks={"after_tool": [ToolLoopDetectionHook()]})

    def test_to_dict_from_dict_roundtrip(self):
        hook = ToolLoopDetectionHook(
            warn_threshold=None,
            stop_threshold=4,
            max_cycle_length=2,
            window_size=12,
            detect_consecutive_errors=False,
            add_final_message=True,
        )
        restored = ToolLoopDetectionHook.from_dict(hook.to_dict())
        assert restored.warn_threshold is None
        assert restored.stop_threshold == 4
        assert restored.max_cycle_length == 2
        assert restored.window_size == 12
        assert restored.detect_consecutive_errors is False
        assert restored.add_final_message is True

    def test_agent_serialization_roundtrip_with_hook(self):
        agent = Agent(
            chat_generator=MockChatGenerator(),
            tools=[search],
            hooks={"before_llm": [ToolLoopDetectionHook(warn_threshold=2, stop_threshold=4, add_final_message=True)]},
        )
        restored = Agent.from_dict(agent.to_dict())
        assert len(restored.hooks["before_llm"]) == 1
        restored_hook = restored.hooks["before_llm"][0]
        assert isinstance(restored_hook, ToolLoopDetectionHook)
        assert restored_hook.warn_threshold == 2
        assert restored_hook.stop_threshold == 4
        assert restored_hook.add_final_message is True

    def test_warn_is_idempotent_on_same_state(self):
        tc = ToolCall(tool_name="search", arguments={"query": "dup"})
        state = _state_with_messages(
            [ChatMessage.from_user("hi"), *_tool_step_messages([tc]), *_tool_step_messages([tc])]
        )
        hook = ToolLoopDetectionHook(warn_threshold=2, stop_threshold=3)

        hook.run(state)
        count_after_first = len(state.data["messages"])
        assert state.data["messages"][-1].meta[_SYSTEM_REMINDER_META_KEY] is True
        assert _LOOP_WARNING_META_KEY in state.data["messages"][-1].meta

        hook.run(state)
        assert len(state.data["messages"]) == count_after_first

    def test_warn_threshold_none_skips_reminder_and_stops_directly(self, caplog):
        tc = ToolCall(tool_name="search", arguments={"query": "x"})
        hook = ToolLoopDetectionHook(warn_threshold=None, stop_threshold=3, add_final_message=True)

        state_2 = _state_with_messages(
            [ChatMessage.from_user("hi"), *_tool_step_messages([tc]), *_tool_step_messages([tc])]
        )
        hook.run(state_2)
        assert state_2.data.get("stop_run") is None
        assert not any(_SYSTEM_REMINDER_META_KEY in m.meta for m in state_2.data["messages"])

        state_3 = _state_with_messages(
            [
                ChatMessage.from_user("hi"),
                *_tool_step_messages([tc]),
                *_tool_step_messages([tc]),
                *_tool_step_messages([tc]),
            ]
        )
        with caplog.at_level(logging.WARNING):
            hook.run(state_3)
        assert state_3.data["stop_run"] == _EXIT_REASON_TOOL_LOOP
        assert state_3.data["messages"][-1].is_from(ChatRole.ASSISTANT)
        assert "tool-calling loop was detected" in (state_3.data["messages"][-1].text or "")
        assert "exact_repeat" in caplog.text

    def test_agent_warns_then_stops_on_exact_repeat(self, monkeypatch):
        agent = Agent(
            chat_generator=MockChatGenerator(),
            tools=[search],
            max_agent_steps=10,
            hooks={"before_llm": [ToolLoopDetectionHook(warn_threshold=2, stop_threshold=3)]},
        )
        agent.warm_up()
        seen_last_messages: list[ChatMessage] = []

        def fake_run(messages: list[ChatMessage], **kwargs: Any) -> dict[str, Any]:
            seen_last_messages.append(messages[-1])
            return {"replies": [ChatMessage.from_assistant(tool_calls=[ToolCall("search", {"query": "loop"})])]}

        mock_run = MagicMock(side_effect=fake_run)
        monkeypatch.setattr(agent.chat_generator, "run", mock_run)
        result = agent.run(messages=[ChatMessage.from_user("start")])

        assert result["exit_reason"] == _EXIT_REASON_TOOL_LOOP
        assert result["step_count"] == 3
        assert mock_run.call_count == 3
        assert seen_last_messages[2].is_from(ChatRole.USER)
        assert seen_last_messages[2].meta.get(_SYSTEM_REMINDER_META_KEY) is True
        assert "Loop detection warning:" in (seen_last_messages[2].text or "")

    def test_agent_recovers_after_loop_warning(self, monkeypatch):
        agent = Agent(
            chat_generator=MockChatGenerator(),
            tools=[search],
            max_agent_steps=10,
            hooks={"before_llm": [ToolLoopDetectionHook(warn_threshold=2, stop_threshold=3)]},
        )
        agent.warm_up()

        def fake_run(messages: list[ChatMessage], **kwargs: Any) -> dict[str, Any]:
            if messages[-1].meta.get(_SYSTEM_REMINDER_META_KEY):
                return {"replies": [ChatMessage.from_assistant("Recovered with final answer.")]}
            return {"replies": [ChatMessage.from_assistant(tool_calls=[ToolCall("search", {"query": "loop"})])]}

        monkeypatch.setattr(agent.chat_generator, "run", MagicMock(side_effect=fake_run))
        result = agent.run(messages=[ChatMessage.from_user("start")])

        assert result["exit_reason"] == "text"
        assert result["last_message"].text == "Recovered with final answer."
        assert result["step_count"] == 3

    def test_agent_stops_on_ping_pong_cycle(self, monkeypatch):
        agent = Agent(
            chat_generator=MockChatGenerator(),
            tools=[search, fetch],
            max_agent_steps=12,
            hooks={"before_llm": [ToolLoopDetectionHook(warn_threshold=2, stop_threshold=3, max_cycle_length=2)]},
        )
        agent.warm_up()
        call_idx = 0

        def fake_run(messages: list[ChatMessage], **kwargs: Any) -> dict[str, Any]:
            nonlocal call_idx
            tc = (
                ToolCall("search", {"query": "ping"})
                if call_idx % 2 == 0
                else ToolCall("fetch", {"url": "https://pong.test"})
            )
            call_idx += 1
            return {"replies": [ChatMessage.from_assistant(tool_calls=[tc])]}

        monkeypatch.setattr(agent.chat_generator, "run", MagicMock(side_effect=fake_run))
        result = agent.run(messages=[ChatMessage.from_user("start")])

        assert result["exit_reason"] == _EXIT_REASON_TOOL_LOOP
        assert result["step_count"] == 6

    def test_agent_stops_on_consecutive_tool_errors(self, monkeypatch):
        agent = Agent(
            chat_generator=MockChatGenerator(),
            tools=[flaky],
            max_agent_steps=10,
            hooks={"before_llm": [ToolLoopDetectionHook(warn_threshold=2, stop_threshold=3)]},
        )
        agent.warm_up()
        attempt = 0

        def fake_run(messages: list[ChatMessage], **kwargs: Any) -> dict[str, Any]:
            nonlocal attempt
            attempt += 1
            return {"replies": [ChatMessage.from_assistant(tool_calls=[ToolCall("flaky", {"attempt": attempt})])]}

        monkeypatch.setattr(agent.chat_generator, "run", MagicMock(side_effect=fake_run))
        result = agent.run(messages=[ChatMessage.from_user("run flaky")])

        assert result["exit_reason"] == _EXIT_REASON_TOOL_LOOP
        assert result["step_count"] == 3

    @pytest.mark.asyncio
    async def test_shared_hook_is_isolated_across_concurrent_async_runs(self, monkeypatch):
        shared_hook = ToolLoopDetectionHook(warn_threshold=2, stop_threshold=3)
        agent = Agent(
            chat_generator=MockChatGenerator(), tools=[search], max_agent_steps=10, hooks={"before_llm": [shared_hook]}
        )
        await agent.warm_up_async()

        async def fake_run_async(messages: list[ChatMessage], **kwargs: Any) -> dict[str, Any]:
            await asyncio.sleep(0)
            prompt = messages[0].text or ""
            if prompt == "looping":
                return {"replies": [ChatMessage.from_assistant(tool_calls=[ToolCall("search", {"query": "same"})])]}
            if len(messages) == 1:
                return {"replies": [ChatMessage.from_assistant(tool_calls=[ToolCall("search", {"query": "once"})])]}
            return {"replies": [ChatMessage.from_assistant("healthy done")]}

        monkeypatch.setattr(agent.chat_generator, "run_async", fake_run_async)
        loop_res, healthy_res = await asyncio.gather(
            agent.run_async(messages=[ChatMessage.from_user("looping")]),
            agent.run_async(messages=[ChatMessage.from_user("healthy")]),
        )

        assert loop_res["exit_reason"] == _EXIT_REASON_TOOL_LOOP
        assert loop_res["step_count"] == 3
        assert healthy_res["exit_reason"] == "text"
        assert healthy_res["last_message"].text == "healthy done"

    def test_agent_without_hook_behavior_unchanged(self, monkeypatch):
        agent = Agent(chat_generator=MockChatGenerator(), tools=[search], max_agent_steps=4)
        agent.warm_up()
        monkeypatch.setattr(
            agent.chat_generator,
            "run",
            MagicMock(
                return_value={
                    "replies": [ChatMessage.from_assistant(tool_calls=[ToolCall("search", {"query": "same"})])]
                }
            ),
        )
        result = agent.run(messages=[ChatMessage.from_user("start")])
        assert result["exit_reason"] == "max_agent_steps"
        assert result["step_count"] == 4
