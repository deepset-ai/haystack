# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack.components.agents.state import State
from haystack.hooks.invocation import _run_hooks, _run_hooks_async


class RecordingHook:
    """Sync-only hook (no `run_async`), to exercise the async fallback path."""

    def __init__(self, label: str, log: list) -> None:
        self.label = label
        self.log = log

    def run(self, state: State) -> None:
        self.log.append(("run", self.label))


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
        _run_hooks(hooks, "before_llm", State(schema={}))
        assert log == [("run", "a"), ("run", "b")]

    def test_only_runs_the_given_hook_point(self):
        log: list = []
        hooks = {"before_llm": [RecordingHook("a", log)], "on_exit": [RecordingHook("b", log)]}
        _run_hooks(hooks, "on_exit", State(schema={}))
        assert log == [("run", "b")]

    def test_no_hooks_for_hook_point_is_noop(self):
        _run_hooks({}, "before_llm", State(schema={}))  # does not raise


class TestRunHooksAsync:
    @pytest.mark.asyncio
    async def test_awaits_run_async_when_present(self):
        log: list = []
        await _run_hooks_async({"before_llm": [AsyncRecordingHook("a", log)]}, "before_llm", State(schema={}))
        assert log == [("run_async", "a")]

    @pytest.mark.asyncio
    async def test_falls_back_to_run_when_no_run_async(self):
        log: list = []
        await _run_hooks_async({"before_llm": [RecordingHook("a", log)]}, "before_llm", State(schema={}))
        assert log == [("run", "a")]

    @pytest.mark.asyncio
    async def test_runs_in_order_mixing_sync_and_async(self):
        log: list = []
        hooks = {"before_llm": [AsyncRecordingHook("a", log), RecordingHook("b", log)]}
        await _run_hooks_async(hooks, "before_llm", State(schema={}))
        assert log == [("run_async", "a"), ("run", "b")]
