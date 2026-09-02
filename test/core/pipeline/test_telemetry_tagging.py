# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import haystack.telemetry._telemetry as telemetry_module
from haystack import Pipeline
from haystack.core.serialization import generate_qualified_class_name
from haystack.testing.sample_components import AddFixedValue


def test_pipeline_run_reports_tagged_event_name(monkeypatch):
    """
    Covers the autouse fixture in conftest.py: Pipeline runs in this test suite should report a
    "(tests)"-tagged event name instead of the production "Pipeline run (3.x)" one.

    Uses a real Telemetry instance (not a mocked telemetry object) so this exercises the actual
    Telemetry.send_event() code path end-to-end - only the final network call to PostHog
    (posthog.capture) is stubbed out, so nothing is ever sent.
    """
    monkeypatch.setattr(telemetry_module, "telemetry", telemetry_module.Telemetry())
    mock_capture = MagicMock()
    monkeypatch.setattr(telemetry_module.posthog, "capture", mock_capture)

    pipe = Pipeline()
    pipe.add_component("add", AddFixedValue())
    pipe.run({"add": {"value": 1}})

    mock_capture.assert_called_once()
    assert mock_capture.call_args.kwargs["event"] == "Pipeline run (3.x) (tests)"

    properties = mock_capture.call_args.kwargs["properties"]
    assert properties["pipeline_type"] == generate_qualified_class_name(Pipeline)
    assert properties["components"] == {generate_qualified_class_name(AddFixedValue): [{"name": "add"}]}
