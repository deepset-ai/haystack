# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import inspect
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from haystack import component, tracing
from haystack.testing.test_utils import set_all_seeds
from test.tracing.utils import SpyingTracer

set_all_seeds(0)


def pytest_collection_modifyitems(items):
    """Automatically apply the 'no_leaks' marker to all tests for pyleak instrumentation."""
    marker = pytest.mark.no_leaks
    for item in items:
        item.add_marker(marker)


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_call(item: pytest.Function):
    """Fix pyleak's no_leaks marker for class-based test methods.

    pyleak's pytest plugin replaces item.obj with a wrapped function, but for class-based
    tests the method is retrieved from the instance, not from item.obj. This hook patches
    the method directly on the instance so leak detection actually runs.

    We run tryfirst and strip the marker so pyleak's own hook skips the test (we handle it).

    Reference: https://github.com/deepankarm/pyleak/issues/17
    """
    try:
        from pyleak.combined import CombinedLeakDetector, PyLeakConfig
        from pyleak.utils import CallerContext
    except ImportError:
        yield
        return

    marker = item.get_closest_marker("no_leaks")
    if not marker:
        yield
        return

    instance = getattr(item, "instance", None)
    if instance is None:
        # Not a class-based test, let pyleak's default plugin handle it
        yield
        return

    test_func = getattr(item, "obj", None) or getattr(item, "function", None)
    if test_func is None:
        yield
        return

    is_async = inspect.iscoroutinefunction(test_func)

    # Parse marker arguments (same logic as pyleak's should_monitor_test)
    marker_args: dict[str, Any] = {}
    if marker.args:
        for arg in marker.args:
            if arg in ("tasks", "threads", "blocking"):
                marker_args[arg] = True
            elif arg == "all":
                marker_args.update({"tasks": True, "threads": True, "blocking": True})
    if marker.kwargs:
        marker_args.update(marker.kwargs)
    if not marker_args:
        marker_args = {"tasks": True, "threads": True, "blocking": True}

    config = PyLeakConfig.from_marker_args(marker_args)
    caller_context = CallerContext(
        filename=str(item.fspath) if item.fspath else "<unknown>", name=item.name, lineno=None
    )

    # item.obj is a bound method for class-based tests. We wrap it and replace item.obj
    # so that both pytest-asyncio and standard pytest pick up our wrapper.
    original_obj = item.obj

    if is_async:

        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            detector = CombinedLeakDetector(config=config, is_async=True, caller_context=caller_context)
            async with detector:
                return await original_obj(*args, **kwargs)

        item.obj = async_wrapper
    else:

        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            detector = CombinedLeakDetector(config=config, is_async=False, caller_context=caller_context)
            with detector:
                return original_obj(*args, **kwargs)

        item.obj = sync_wrapper

    # Strip the no_leaks marker so pyleak's own hook doesn't double-wrap and break
    saved_markers = item.own_markers[:]
    item.own_markers = [m for m in item.own_markers if m.name != "no_leaks"]

    try:
        yield
    finally:
        item.own_markers = saved_markers
        item.obj = original_obj


# Tracing is disable by default to avoid failures in CI
tracing.disable_tracing()


@pytest.fixture()
def waiting_component():
    @component
    class Waiter:
        @component.output_types(waited_for=int)
        def run(self, wait_for: int) -> dict[str, int]:
            time.sleep(wait_for)
            return {"waited_for": wait_for}

        @component.output_types(waited_for=int)
        async def run_async(self, wait_for: int) -> dict[str, int]:
            await asyncio.sleep(wait_for)
            return {"waited_for": wait_for}

    return Waiter


@pytest.fixture()
def mock_tokenizer():
    """
    Tokenizes the string by splitting on spaces.
    """
    tokenizer = Mock()
    tokenizer.encode = lambda text: text.split()
    tokenizer.decode = lambda tokens: " ".join(tokens)  # noqa: PLW0108
    return tokenizer


@pytest.fixture()
def test_files_path():
    return Path(__file__).parent / "test_files"


@pytest.fixture(autouse=True)
def request_blocker(request: pytest.FixtureRequest, monkeypatch):
    """
    This fixture is applied automatically to all tests.
    Those that are not marked as integration will have the requests module
    monkeypatched to avoid making HTTP requests by mistake.
    """
    marker = request.node.get_closest_marker("integration")
    if marker is not None:
        return

    def urlopen_mock(self, method, url, *args, **kwargs):
        raise RuntimeError(f"The test was about to {method} {self.scheme}://{self.host}{url}")

    monkeypatch.setattr("urllib3.connectionpool.HTTPConnectionPool.urlopen", urlopen_mock)


@pytest.fixture()
def spying_tracer() -> Generator[SpyingTracer, None, None]:
    tracer = SpyingTracer()
    tracing.enable_tracing(tracer)
    tracer.is_content_tracing_enabled = True

    yield tracer

    # Make sure to disable tracing after the test to avoid affecting other tests
    tracing.disable_tracing()


@pytest.fixture()
def base64_image_string():
    return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII="


@pytest.fixture()
def base64_pdf_string(test_files_path):
    with open(test_files_path / "pdf" / "sample_pdf_3.pdf", "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


@pytest.fixture
def del_hf_env_vars(monkeypatch):
    """
    Delete Hugging Face environment variables for tests.

    Prevents passing empty tokens to Hugging Face, which would cause API calls to fail.
    This is particularly relevant for PRs opened from forks, where secrets are not available
    and empty environment variables might be set instead of being removed.

    See https://github.com/deepset-ai/haystack/issues/8811 for more details.
    """
    monkeypatch.delenv("HF_API_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
