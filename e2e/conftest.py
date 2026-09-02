# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import Mock

import pytest

import haystack.core.pipeline.pipeline as pipeline_module
import haystack.telemetry._telemetry as telemetry_module
from haystack.telemetry._telemetry import pipeline_running, send_telemetry
from haystack.testing.test_utils import set_all_seeds

set_all_seeds(0)


@pytest.fixture(autouse=True)
def block_telemetry_network_calls(monkeypatch):
    """
    Force `telemetry` to be a real, truthy Telemetry instance (regardless of the ambient
    HAYSTACK_TELEMETRY_ENABLED setting) so every Pipeline.run()/run_async() in the e2e suite still
    exercises the real @send_telemetry / Telemetry.send_event() code path - catching any real
    breakage there - but stub out `posthog.capture`, the only place that code path ever touches
    the network, so e2e runs never send real telemetry data or connect to PostHog. E2e tests
    intentionally hit real external services (e.g. OpenAI), but PostHog telemetry isn't what's
    under test and shouldn't pollute production analytics with events from CI runs.
    """
    monkeypatch.setattr(telemetry_module, "telemetry", telemetry_module.Telemetry())
    monkeypatch.setattr(telemetry_module.posthog, "capture", Mock())


@pytest.fixture(autouse=True)
def tag_pipeline_telemetry_as_test(monkeypatch):
    """
    Pipeline.run()/run_async() report a "Pipeline run (3.x)" event to Posthog on every call. Tag
    events triggered by the e2e suite with a distinct name so they can be told apart from real
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


@pytest.fixture
def samples_path():
    return Path(__file__).parent / "samples"


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
