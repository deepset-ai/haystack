# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

import haystack.core.pipeline.pipeline as pipeline_module
import haystack.telemetry._telemetry as telemetry_module
from haystack.telemetry._telemetry import pipeline_running, send_telemetry


@pytest.fixture(autouse=True)
def block_telemetry_network_calls(monkeypatch):
    """
    Force `telemetry` to be a real, truthy Telemetry instance (regardless of the ambient
    HAYSTACK_TELEMETRY_ENABLED setting) so every Pipeline.run()/run_async() in this test suite
    still exercises the real @send_telemetry / Telemetry.send_event() code path - catching any
    real breakage there - but stub out `posthog.capture`, the only place that code path ever
    touches the network, so no test ever actually sends data or connects to PostHog.
    """
    monkeypatch.setattr(telemetry_module, "telemetry", telemetry_module.Telemetry())
    monkeypatch.setattr(telemetry_module.posthog, "capture", MagicMock())


@pytest.fixture(autouse=True)
def tag_pipeline_telemetry_as_test(monkeypatch):
    """
    Pipeline.run()/run_async() report a "Pipeline run (3.x)" event to Posthog on every call. Tag
    events triggered by this test suite with a distinct name so they can be told apart from real
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
