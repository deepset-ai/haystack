# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import datetime
import logging
from unittest.mock import Mock, patch

import pytest

from haystack import Pipeline, component
from haystack.telemetry._telemetry import pipeline_running
from haystack.utils.auth import Secret, TokenSecret


@patch("haystack.telemetry._telemetry.telemetry")
def test_pipeline_running(telemetry):
    telemetry.send_event = Mock()

    @component
    class Component:
        def _get_telemetry_data(self):
            return {"key": "values"}

        @component.output_types(value=int)
        def run(self):
            pass

    pipe = Pipeline()
    pipe.add_component("component", Component())
    pipeline_running(pipe)

    # First run is always sent
    telemetry.send_event.assert_called_once_with(
        "Pipeline run (2.x)",
        {
            "pipeline_id": str(id(pipe)),
            "runs": 1,
            "components": {"test.test_telemetry.Component": [{"name": "component", "key": "values"}]},
        },
    )

    # Running again before one minute has passed should not send another event
    telemetry.send_event.reset_mock()
    pipeline_running(pipe)
    telemetry.send_event.assert_not_called()

    # Set the last telemetry sent time to pretend one minute has passed
    pipe._last_telemetry_sent = pipe._last_telemetry_sent - datetime.timedelta(minutes=1)

    telemetry.send_event.reset_mock()
    pipeline_running(pipe)
    telemetry.send_event.assert_called_once_with(
        "Pipeline run (2.x)",
        {
            "pipeline_id": str(id(pipe)),
            "runs": 3,
            "components": {"test.test_telemetry.Component": [{"name": "component", "key": "values"}]},
        },
    )


@patch("haystack.telemetry._telemetry.telemetry")
def test_pipeline_running_with_non_serializable_component(telemetry):
    telemetry.send_event = Mock()

    @component
    class Component:
        def __init__(self, api_key: Secret = TokenSecret("api_key")):
            self.api_key = api_key

        def _get_telemetry_data(self):
            return {"key": "values"}

        @component.output_types(value=int)
        def run(self):
            pass

    pipe = Pipeline()
    pipe.add_component("component", Component())
    pipeline_running(pipe)
    telemetry.send_event.assert_called_once_with(
        "Pipeline run (2.x)",
        {
            "pipeline_id": str(id(pipe)),
            "runs": 1,
            "components": {"test.test_telemetry.Component": [{"name": "component", "key": "values"}]},
        },
    )


def test_pipeline_running_with_non_dict_telemetry_data(caplog):
    @component
    class Component:
        def __init__(self, api_key: Secret = TokenSecret("api_key")):
            self.api_key = api_key

        # telemetry data should be a dictionary but is a list
        def _get_telemetry_data(self):
            return ["values"]

        @component.output_types(value=int)
        def run(self):
            pass

    pipe = Pipeline()
    pipe.add_component("my_component", Component())
    with caplog.at_level(logging.DEBUG):
        pipeline_running(pipe)
        assert "TypeError: Telemetry data for component my_component must be a dictionary" in caplog.text
