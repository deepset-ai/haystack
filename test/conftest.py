# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import time
from pathlib import Path
from typing import Dict, Generator
from unittest.mock import Mock

import pytest

from haystack import component, tracing
from haystack.core.pipeline.breakpoint import load_pipeline_snapshot
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
        def run(self, wait_for: int) -> Dict[str, int]:
            time.sleep(wait_for)
            return {"waited_for": wait_for}

        @component.output_types(waited_for=int)
        async def run_async(self, wait_for: int) -> Dict[str, int]:
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
    tokenizer.decode = lambda tokens: " ".join(tokens)
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


def load_and_resume_pipeline_snapshot(pipeline, output_directory: Path, component_name: str, data: Dict = None) -> Dict:
    """
    Utility function to load and resume pipeline snapshot from a breakpoint file.

    :param pipeline: The pipeline instance to resume
    :param output_directory: Directory containing the breakpoint files
    :param component_name: Component name to look for in breakpoint files
    :param data: Data to pass to the pipeline run (defaults to empty dict)

    :returns:
        Dict containing the pipeline run results

    :raises:
        ValueError: If no breakpoint file is found for the given component
    """
    data = data or {}
    all_files = list(output_directory.glob("*"))
    file_found = False

    for full_path in all_files:
        f_name = Path(full_path).name
        if str(f_name).startswith(component_name):
            pipeline_snapshot = load_pipeline_snapshot(full_path)
            return pipeline.run(data=data, pipeline_snapshot=pipeline_snapshot)

    if not file_found:
        msg = f"No files found for {component_name} in {output_directory}."
        raise ValueError(msg)

@pytest.fixture()
def base64_image_string():
    return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII="
