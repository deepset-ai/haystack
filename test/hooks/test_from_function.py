# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack.components.agents.state import State
from haystack.dataclasses import ChatMessage
from haystack.hooks import FunctionHook, hook


def append_system(state: State) -> None:
    state.set("messages", [ChatMessage.from_system("from sync hook")])


async def append_system_async(state: State) -> None:
    state.set("messages", [ChatMessage.from_system("from async hook")])


def append_system_postponed_annotation(state: "State") -> None:
    state.set("messages", [ChatMessage.from_system("from postponed annotation hook")])


class TestHookDecorator:
    def test_wraps_sync_function(self):
        wrapped = hook(append_system)
        assert isinstance(wrapped, FunctionHook)
        assert wrapped.function is append_system
        assert wrapped.async_function is None

    def test_wraps_async_function(self):
        wrapped = hook(append_system_async)
        assert isinstance(wrapped, FunctionHook)
        assert wrapped.function is None
        assert wrapped.async_function is append_system_async

    def test_run_invokes_sync_function(self):
        state = State(schema={})
        hook(append_system).run(state)
        assert [m.text for m in state.data["messages"]] == ["from sync hook"]

    def test_wraps_function_with_postponed_state_annotation(self):
        wrapped = hook(append_system_postponed_annotation)
        state = State(schema={})
        wrapped.run(state)
        assert [m.text for m in state.data["messages"]] == ["from postponed annotation hook"]

    def test_run_on_async_only_raises(self):
        with pytest.raises(RuntimeError):
            hook(append_system_async).run(State(schema={}))


class TestFunctionHookConstruction:
    def test_requires_at_least_one_function(self):
        with pytest.raises(ValueError):
            FunctionHook()

    def test_sync_slot_rejects_coroutine_function(self):
        with pytest.raises(ValueError):
            FunctionHook(function=append_system_async)

    def test_async_slot_rejects_regular_function(self):
        with pytest.raises(ValueError):
            FunctionHook(async_function=append_system)

    def test_rejects_function_without_a_parameter(self):
        def no_params() -> None:
            pass

        with pytest.raises(ValueError):
            FunctionHook(function=no_params)

    def test_rejects_function_with_extra_parameters(self):
        def two_params(state: State, extra: int) -> None:
            pass

        with pytest.raises(ValueError):
            FunctionHook(function=two_params)

    def test_rejects_unannotated_parameter(self):
        def unannotated(state) -> None:
            pass

        with pytest.raises(ValueError):
            FunctionHook(function=unannotated)

    def test_both_functions(self):
        h = FunctionHook(function=append_system, async_function=append_system_async)
        state = State(schema={})
        h.run(state)
        assert [m.text for m in state.data["messages"]] == ["from sync hook"]


class TestFunctionHookSerde:
    def test_to_dict_sync(self):
        data = hook(append_system).to_dict()
        assert data == {
            "type": "haystack.hooks.from_function.FunctionHook",
            "init_parameters": {"function": "test.hooks.test_from_function.append_system", "async_function": None},
        }

    def test_roundtrip_sync(self):
        restored = FunctionHook.from_dict(hook(append_system).to_dict())
        state = State(schema={})
        restored.run(state)
        assert [m.text for m in state.data["messages"]] == ["from sync hook"]

    def test_roundtrip_async(self):
        restored = FunctionHook.from_dict(hook(append_system_async).to_dict())
        assert restored.function is None
        assert restored.async_function is append_system_async

    def test_roundtrip_both(self):
        restored = FunctionHook.from_dict(
            FunctionHook(function=append_system, async_function=append_system_async).to_dict()
        )
        assert restored.function is append_system
        assert restored.async_function is append_system_async


class TestFunctionHookAsync:
    @pytest.mark.asyncio
    async def test_run_async_awaits_async_function(self):
        state = State(schema={})
        await hook(append_system_async).run_async(state)
        assert [m.text for m in state.data["messages"]] == ["from async hook"]

    @pytest.mark.asyncio
    async def test_run_async_falls_back_to_sync_function(self):
        state = State(schema={})
        await hook(append_system).run_async(state)
        assert [m.text for m in state.data["messages"]] == ["from sync hook"]
