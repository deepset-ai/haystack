# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import time
from collections.abc import Generator
from pathlib import Path
from unittest.mock import Mock

import pytest

import haystack.core.pipeline.pipeline as pipeline_module
import haystack.telemetry._telemetry as telemetry_module
from haystack import component, tracing
from haystack.core.serialization import allow_deserialization_module
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.telemetry._telemetry import pipeline_running, send_telemetry
from haystack.testing.test_utils import set_all_seeds
from test.tracing.utils import EagerSpyingTracer, SpyingTracer

set_all_seeds(0)


# Tracing is disable by default to avoid failures in CI
tracing.disable_tracing()


# Tests legitimately deserialize callables/components/types from a handful of modules that aren't
# part of the default Haystack allowlist. We extend the allowlist explicitly.
#
# Tests that exercise the rejection path themselves install a clean context (and clear the
# process-wide patterns); see `test/core/test_serialization_security.py`.
for _pattern in (
    "test_*",  # top-level `test_<name>` modules (pytest rootdir-level files)
    "*.test_*",  # `<subdir>.test_<name>` modules (pytest treats sub-packages this way)
    "test.*",  # modules inside the proper `test` package (with __init__.py)
    "pydantic",  # pydantic models used in base-serialization tests
    "httpx",  # used in callable-serialization tests
):
    allow_deserialization_module(_pattern)


@pytest.fixture()
def in_memory_doc_store():
    store = InMemoryDocumentStore()
    yield store
    store.shutdown()


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


@pytest.fixture(autouse=True)
def block_telemetry_network_calls(monkeypatch):
    """
    Force `telemetry` to be a real, truthy Telemetry instance (regardless of the ambient
    HAYSTACK_TELEMETRY_ENABLED setting) so every Pipeline.run()/run_async() in the test suite
    (directly, or indirectly via SuperComponent, PipelineTool, Agent, retrievers, etc.) still
    exercises the real @send_telemetry / Telemetry.send_event() code path - catching any real
    breakage there - but stub out `posthog.capture`, the only place that code path ever touches
    the network, so no test ever actually sends data or connects to PostHog.
    """
    monkeypatch.setattr(telemetry_module, "telemetry", telemetry_module.Telemetry())
    monkeypatch.setattr(telemetry_module.posthog, "capture", Mock())


@pytest.fixture(autouse=True)
def tag_pipeline_telemetry_as_test(monkeypatch):
    """
    Pipeline.run()/run_async() report a "Pipeline run (3.x)" event to Posthog on every call. Tag
    events triggered by the test suite with a distinct name so they can be told apart from real
    usage, without changing haystack.telemetry.pipeline_running itself.

    `pipeline_running.__wrapped__` is the raw, undecorated function stashed there by
    `functools.wraps` inside the `@send_telemetry` decorator - reusing it (instead of
    reimplementing its logic here) means this stays in sync with the real function automatically.
    """

    @send_telemetry
    def test_pipeline_running(pipeline):
        result = pipeline_running.__wrapped__(pipeline)
        if result is None:
            return None
        event_name, event_properties = result
        return f"{event_name} (tests)", event_properties

    monkeypatch.setattr(pipeline_module, "pipeline_running", test_pipeline_running)


@pytest.fixture()
def spying_tracer() -> Generator[SpyingTracer, None, None]:
    tracer = SpyingTracer()
    tracing.enable_tracing(tracer)
    tracer.is_content_tracing_enabled = True

    yield tracer

    # Make sure to disable tracing after the test to avoid affecting other tests
    tracing.disable_tracing()


@pytest.fixture()
def eager_spying_tracer() -> Generator[EagerSpyingTracer, None, None]:
    # Coerces tags when set, mirroring real backends. Content tracing is left to the test to toggle.
    tracer = EagerSpyingTracer()
    tracing.enable_tracing(tracer)

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
