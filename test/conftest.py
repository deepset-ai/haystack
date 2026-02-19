# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import time
from pathlib import Path
from typing import Generator
from unittest.mock import Mock

import pytest

from haystack import component, tracing
from haystack.testing.test_utils import set_all_seeds
from test.tracing.utils import SpyingTracer

set_all_seeds(0)

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
