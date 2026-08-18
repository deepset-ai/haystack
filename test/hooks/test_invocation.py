# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import functools
import threading

import pytest

from haystack.components.agents.state import State
from haystack.hooks import FunctionHook, hook
from haystack.hooks.invocation import _run_hooks, _run_hooks_async


@hook
def traced_function_hook(state: State) -> None:
    state.set(key="messages", value=[])


def plain_function_hook(state: State) -> None:
    pass


class CallableFunctionHook:
    def __call__(self, state: State) -> None:
        pass


class RecordingHook:
    """Sync-only hook (no `run_async`), to exercise the async fallback path."""

    def __init__(self, label: str, log: list) -> None:
        self.label = label
        self.log = log

    def run(self, state: State) -> None:
        self.log.append(("run", self.label))


class ThreadRecordingHook:
    def __init__(self) -> None:
        self.thread_id: int | None = None

    def run(self, state: State) -> None:
        self.thread_id = threading.get_ident()


class AsyncRecordingHook:
    def __init__(self, label: str, log: list) -> None:
        self.label = label
        self.log = log

    def run(self, state: State) -> None:
        self.log.append(("run", self.label))

    async def run_async(self, state: State) -> None:
        self.log.append(("run_async", self.label))


class TestRunHooks:
    def test_runs_all_hooks_for_hook_point_in_order(self):
        log: list = []
        hooks = {"before_llm": [RecordingHook("a", log), RecordingHook("b", log)]}
        _run_hooks(hooks=hooks, hook_point="before_llm", state=State(schema={}))
        assert log == [("run", "a"), ("run", "b")]

    def test_only_runs_the_given_hook_point(self):
        log: list = []
        hooks = {"before_llm": [RecordingHook("a", log)], "on_exit": [RecordingHook("b", log)]}
        _run_hooks(hooks=hooks, hook_point="on_exit", state=State(schema={}))
        assert log == [("run", "b")]

    def test_no_hooks_for_hook_point_is_noop(self):
        _run_hooks(hooks={}, hook_point="before_llm", state=State(schema={}))  # does not raise

    def test_traces_each_hook_invocation_as_a_sibling(self, spying_tracer):
        log: list = []
        hooks = {"before_llm": [RecordingHook(label="a", log=log), RecordingHook(label="b", log=log)]}
        with spying_tracer.trace(operation_name="parent") as parent_span:
            _run_hooks(hooks=hooks, hook_point="before_llm", state=State(schema={}))
        hook_spans = [span for span in spying_tracer.spans if span.operation_name == "haystack.agent.hook"]
        assert len(hook_spans) == 2
        assert all(span.parent_span is parent_span for span in hook_spans)
        assert all(
            span.tags
            == {
                "haystack.agent.hook.point": "before_llm",
                "haystack.agent.hook.name": "RecordingHook",
                "haystack.agent.hook.type": "test.hooks.test_invocation.RecordingHook",
            }
            for span in hook_spans
        )

    def test_function_hook_span_identifies_wrapped_function(self, spying_tracer):
        _run_hooks(hooks={"before_run": [traced_function_hook]}, hook_point="before_run", state=State(schema={}))
        span = spying_tracer.spans[0]
        assert span.operation_name == "haystack.agent.hook"
        assert span.tags == {
            "haystack.agent.hook.point": "before_run",
            "haystack.agent.hook.name": "test.hooks.test_invocation.traced_function_hook",
            "haystack.agent.hook.type": "haystack.hooks.from_function.FunctionHook",
        }

    @pytest.mark.parametrize(
        "function", [functools.partial(plain_function_hook), CallableFunctionHook()], ids=["partial", "callable-object"]
    )
    def test_function_hook_name_falls_back_when_callable_has_no_name(self, spying_tracer, function):
        # We don't support serialization of partials or callable-object instances, so the span name falls back to the
        # class name.
        function_hook = FunctionHook(function=function)
        _run_hooks(hooks={"before_run": [function_hook]}, hook_point="before_run", state=State(schema={}))
        assert spying_tracer.spans[0].tags["haystack.agent.hook.name"] == "FunctionHook"

    def test_no_hooks_does_not_create_span(self, spying_tracer):
        _run_hooks(hooks={}, hook_point="before_llm", state=State(schema={}))
        assert spying_tracer.spans == []


class TestRunHooksAsync:
    @pytest.mark.asyncio
    async def test_awaits_run_async_when_present(self):
        log: list = []
        await _run_hooks_async(
            hooks={"before_llm": [AsyncRecordingHook("a", log)]}, hook_point="before_llm", state=State(schema={})
        )
        assert log == [("run_async", "a")]

    @pytest.mark.asyncio
    async def test_falls_back_to_run_when_no_run_async(self):
        log: list = []
        await _run_hooks_async(
            hooks={"before_llm": [RecordingHook("a", log)]}, hook_point="before_llm", state=State(schema={})
        )
        assert log == [("run", "a")]

    @pytest.mark.asyncio
    async def test_falls_back_to_run_in_worker_thread(self):
        hook = ThreadRecordingHook()
        event_loop_thread_id = threading.get_ident()
        await _run_hooks_async(hooks={"before_llm": [hook]}, hook_point="before_llm", state=State(schema={}))
        assert hook.thread_id is not None
        assert hook.thread_id != event_loop_thread_id

    @pytest.mark.asyncio
    async def test_runs_in_order_mixing_sync_and_async(self):
        log: list = []
        hooks = {"before_llm": [AsyncRecordingHook("a", log), RecordingHook("b", log)]}
        await _run_hooks_async(hooks=hooks, hook_point="before_llm", state=State(schema={}))
        assert log == [("run_async", "a"), ("run", "b")]

    @pytest.mark.asyncio
    async def test_traces_async_hook_invocation(self, spying_tracer):
        log: list = []
        hook_instance = AsyncRecordingHook(label="a", log=log)
        with spying_tracer.trace(operation_name="parent") as parent_span:
            await _run_hooks_async(
                hooks={"after_tool": [hook_instance]}, hook_point="after_tool", state=State(schema={})
            )
        hook_span = spying_tracer.spans[1]
        assert hook_span.operation_name == "haystack.agent.hook"
        assert hook_span.parent_span is parent_span
        assert hook_span.tags == {
            "haystack.agent.hook.point": "after_tool",
            "haystack.agent.hook.name": "AsyncRecordingHook",
            "haystack.agent.hook.type": "test.hooks.test_invocation.AsyncRecordingHook",
        }
