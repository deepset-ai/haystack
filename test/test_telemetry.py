import datetime
from unittest.mock import Mock, patch
import pytest

from haystack.preview import Pipeline, component
from haystack.preview.telemetry._telemetry import pipeline_running


@pytest.mark.unit
@patch("haystack.preview.telemetry._telemetry.telemetry")
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
            "components": {"test_telemetry.Component": [{"name": "component", "key": "values"}]},
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
            "components": {"test_telemetry.Component": [{"name": "component", "key": "values"}]},
        },
    )
