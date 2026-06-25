# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack.components.agents.state import State
from haystack.hooks import FunctionHook, hook
from haystack.hooks.utils import (
    _deserialize_hooks,
    _serialize_hooks,
    _unique_hooks,
    close_hooks,
    close_hooks_async,
    warm_up_hooks,
    warm_up_hooks_async,
)


def noop(state: State) -> None:
    pass


class LifecycleSpy:
    """Hook recording its lifecycle calls."""

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


class WarmOnlyHook:
    """Hook with a sync `warm_up`/`close` but no async variants."""

    def __init__(self) -> None:
        self.warmed = 0
        self.closed = 0

    def run(self, state: State) -> None:
        pass

    def warm_up(self) -> None:
        self.warmed += 1

    def close(self) -> None:
        self.closed += 1


class PlainHook:
    """Hook with no lifecycle methods."""

    def run(self, state: State) -> None:
        pass


class TestUniqueHooks:
    def test_dedupes_by_identity_preserving_order(self):
        a, b = PlainHook(), PlainHook()
        unique = _unique_hooks({"before_llm": [a, b], "on_exit": [a]})
        assert unique == [a, b]

    def test_empty(self):
        assert _unique_hooks({}) == []


class TestSerializeHooks:
    def test_roundtrip_multiple_events(self):
        hooks = {"before_llm": [hook(noop)], "on_exit": [hook(noop)]}
        restored = _deserialize_hooks(_serialize_hooks(hooks))
        assert set(restored) == {"before_llm", "on_exit"}
        assert all(isinstance(h, FunctionHook) for hook_list in restored.values() for h in hook_list)
        assert restored["before_llm"][0].function is noop


class TestWarmUpHooks:
    def test_calls_warm_up_when_present(self):
        spy = LifecycleSpy()
        warm_up_hooks({"before_llm": [spy]})
        assert spy.warmed == 1

    def test_skips_hooks_without_warm_up(self):
        warm_up_hooks({"before_llm": [PlainHook()]})  # does not raise

    def test_reused_hook_warmed_once(self):
        spy = LifecycleSpy()
        warm_up_hooks({"before_llm": [spy], "on_exit": [spy]})
        assert spy.warmed == 1


class TestCloseHooks:
    def test_calls_close_when_present(self):
        spy = LifecycleSpy()
        close_hooks({"before_llm": [spy]})
        assert spy.closed == 1

    def test_skips_hooks_without_close(self):
        close_hooks({"before_llm": [PlainHook()]})  # does not raise


class TestWarmUpHooksAsync:
    @pytest.mark.asyncio
    async def test_prefers_async_warm_up(self):
        spy = LifecycleSpy()
        await warm_up_hooks_async({"before_llm": [spy]})
        assert spy.warmed_async == 1
        assert spy.warmed == 0

    @pytest.mark.asyncio
    async def test_falls_back_to_sync_warm_up(self):
        hook_obj = WarmOnlyHook()
        await warm_up_hooks_async({"before_llm": [hook_obj]})
        assert hook_obj.warmed == 1


class TestCloseHooksAsync:
    @pytest.mark.asyncio
    async def test_prefers_async_close(self):
        spy = LifecycleSpy()
        await close_hooks_async({"before_llm": [spy]})
        assert spy.closed_async == 1
        assert spy.closed == 0

    @pytest.mark.asyncio
    async def test_falls_back_to_sync_close(self):
        hook_obj = WarmOnlyHook()
        await close_hooks_async({"before_llm": [hook_obj]})
        assert hook_obj.closed == 1
