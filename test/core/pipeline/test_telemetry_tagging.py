# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import haystack.telemetry._telemetry as telemetry_module
from haystack import Pipeline
from haystack.testing.sample_components import AddFixedValue


def test_pipeline_run_reports_tagged_event_name(monkeypatch):
    """
    Covers the autouse fixture in conftest.py: Pipeline runs in this test suite should report a
    "(tests)"-tagged event name instead of the production "Pipeline run (3.x)" one.
    """
    mock_telemetry = MagicMock()
    monkeypatch.setattr(telemetry_module, "telemetry", mock_telemetry)

    pipe = Pipeline()
    pipe.add_component("add", AddFixedValue())
    pipe.run({"add": {"value": 1}})

    mock_telemetry.send_event.assert_called_once()
    event_name = mock_telemetry.send_event.call_args[0][0]
    assert event_name == "Pipeline run (3.x) (tests)"
