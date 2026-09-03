# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import inspect
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import pytest

import haystack.core.pipeline.pipeline as pipeline_module
import haystack.telemetry._telemetry as telemetry_module
from haystack.telemetry._telemetry import Telemetry, pipeline_running, send_telemetry

if TYPE_CHECKING:
    from haystack.core.pipeline import Pipeline


@pytest.fixture(scope="session")
def telemetry_instance(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Telemetry]:
    """
    A real `Telemetry` instance whose config file lives outside the developer's home directory.

    Instantiating `Telemetry()` reads `~/.haystack/config.yaml` and creates it - with a fresh telemetry `user_id` -
    when it is missing. Pointing `CONFIG_PATH` at a temporary directory for the whole session keeps the test suite
    from writing that file into the home directory of someone who opted out via `HAYSTACK_TELEMETRY_ENABLED=false`.
    """
    config_path = tmp_path_factory.mktemp("haystack_telemetry") / "config.yaml"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(telemetry_module, "CONFIG_PATH", config_path)
        yield Telemetry()


@pytest.fixture(autouse=True)
def block_telemetry_network_calls(monkeypatch: pytest.MonkeyPatch, telemetry_instance: Telemetry) -> Mock:
    """
    Run the real telemetry code path on every Pipeline.run()/run_async(), but never let it reach PostHog.

    Forces `telemetry` to be a real, truthy Telemetry instance (regardless of the ambient
    HAYSTACK_TELEMETRY_ENABLED setting) so every Pipeline.run()/run_async() in the test suite
    (directly, or indirectly via SuperComponent, PipelineTool, Agent, retrievers, etc.) still
    exercises the real @send_telemetry / Telemetry.send_event() code path - but stubs out
    `posthog.capture`, the only place that code path ever touches the network, so no test ever
    actually sends data or connects to PostHog.

    Returns the `posthog.capture` mock so a test can assert on the event that would have been sent.
    """
    monkeypatch.setattr(telemetry_module, "telemetry", telemetry_instance)
    capture = Mock()
    monkeypatch.setattr(telemetry_module.posthog, "capture", capture)
    return capture


@pytest.fixture(autouse=True)
def tag_pipeline_telemetry_as_test(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Tag the "Pipeline run (3.x)" telemetry event with "(tests)" for every Pipeline.run()/run_async() in the suite.

    Pipeline.run()/run_async() report a "Pipeline run (3.x)" event to Posthog on every call. Tagging
    events triggered by the test suite with a distinct name lets them be told apart from real
    usage, without changing haystack.telemetry.pipeline_running itself.

    `inspect.unwrap(pipeline_running)` returns the raw, undecorated function that `functools.wraps`
    stashes in `__wrapped__` inside the `@send_telemetry` decorator - reusing it (instead of
    reimplementing its logic here) means this stays in sync with the real function automatically.
    """

    @send_telemetry
    def _tagged_pipeline_running(pipeline: "Pipeline") -> tuple[str, dict[str, Any]] | None:
        result = inspect.unwrap(pipeline_running)(pipeline)
        if result is None:
            return None
        event_name, event_properties = result
        return f"{event_name} (tests)", event_properties

    monkeypatch.setattr(pipeline_module, "pipeline_running", _tagged_pipeline_running)
