# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import contextvars
import logging

import pytest

from haystack.utils.async_utils import _run_component_async

_test_context_var: contextvars.ContextVar[str] = contextvars.ContextVar("_test_context_var", default="unset")


class ComponentReadingContextVar:
    def run(self, **kwargs):
        # Read inside the executor thread — only visible if the calling context was copied.
        return {"value": _test_context_var.get()}


class ComponentWithRunAsync:
    def run(self, **kwargs):
        return {"path": "sync", "kwargs": kwargs}

    async def run_async(self, **kwargs):
        return {"path": "async", "kwargs": kwargs}


class ComponentWithoutRunAsync:
    def run(self, **kwargs):
        return {"path": "sync", "kwargs": kwargs}


class ComponentWithNonCallableRunAsync:
    run_async = None

    def run(self, **kwargs):
        return {"path": "sync", "kwargs": kwargs}


class TestRunComponentAsync:
    @pytest.mark.asyncio
    async def test_awaits_run_async_when_available(self):
        component = ComponentWithRunAsync()
        result = await _run_component_async(component, foo="bar", count=1)
        assert result == {"path": "async", "kwargs": {"foo": "bar", "count": 1}}

    @pytest.mark.asyncio
    async def test_falls_back_to_sync_run_when_no_run_async(self):
        component = ComponentWithoutRunAsync()
        result = await _run_component_async(component, foo="bar", count=2)
        assert result == {"path": "sync", "kwargs": {"foo": "bar", "count": 2}}

    @pytest.mark.asyncio
    async def test_falls_back_to_sync_run_when_run_async_not_callable(self):
        component = ComponentWithNonCallableRunAsync()
        result = await _run_component_async(component, foo="baz")
        assert result == {"path": "sync", "kwargs": {"foo": "baz"}}

    @pytest.mark.asyncio
    async def test_emits_debug_log_on_sync_fallback(self, caplog):
        component = ComponentWithoutRunAsync()
        with caplog.at_level(logging.DEBUG):
            await _run_component_async(component)
        assert "does not implement 'run_async'" in caplog.text
        assert "ComponentWithoutRunAsync" in caplog.text

    @pytest.mark.asyncio
    async def test_contextvars_propagate_to_sync_run_in_thread(self):
        # Regression test: contextvars set in the calling async context (e.g. the active tracing span) must be
        # visible inside the sync `run` executed in a thread. `asyncio.to_thread` guarantees this by copying the
        # current context; a plain `loop.run_in_executor` would not.
        component = ComponentReadingContextVar()
        _test_context_var.set("propagated")
        result = await _run_component_async(component)
        assert result == {"value": "propagated"}
