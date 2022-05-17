from pathlib import Path

from unittest.mock import patch, PropertyMock

import pytest

from haystack import telemetry
from haystack.errors import PipelineSchemaError
from haystack.telemetry import (
    NonPrivateParameters,
    send_event,
    enable_writing_events_to_file,
    disable_writing_events_to_file,
    send_custom_event,
    _delete_telemetry_file,
    disable_telemetry,
    enable_telemetry,
    TelemetryFileType,
    _write_telemetry_config,
)


@patch.object(
    NonPrivateParameters, "param_names", return_value=["top_k", "model_name_or_path"], new_callable=PropertyMock
)
def test_private_params_not_tracked(mock_nonprivateparameters):
    params = {"hostname": "private_hostname", "top_k": 2}
    tracked_params = NonPrivateParameters.apply_filter(params)
    expected_params = {"top_k": 2}
    assert tracked_params == expected_params


@patch.object(
    NonPrivateParameters, "param_names", return_value=["top_k", "model_name_or_path"], new_callable=PropertyMock
)
def test_non_private_params_tracked(mock_nonprivateparameters):
    params = {"model_name_or_path": "test-model", "top_k": 2}
    non_private_params = NonPrivateParameters.apply_filter(params)
    assert non_private_params == params


@patch.object(NonPrivateParameters, "param_names", return_value=[], new_callable=PropertyMock)
def test_only_non_private_params(mock_nonprivateparameters):
    non_private_params = NonPrivateParameters.apply_filter({"top_k": 2})
    assert non_private_params == {}


@pytest.mark.integration
@patch("posthog.capture")
@patch.object(
    NonPrivateParameters,
    "param_names",
    return_value=["top_k", "model_name_or_path", "add_isolated_node_eval"],
    new_callable=PropertyMock,
)
# patches are applied in bottom-up order, which is why mock_nonprivateparameters is the first parameter and mock_posthog_capture is the second
def test_send_event_via_decorator(mock_nonprivateparameters, mock_posthog_capture):
    class TestClass:
        @send_event
        def run(self, add_isolated_node_eval: bool = False):
            pass

    test_class = TestClass()
    test_class.run(add_isolated_node_eval=True)
    # todo replace [1] with .kwargs when moving from python 3.7 to 3.8 in CI
    assert mock_posthog_capture.call_args[1]["event"] == "TestClass.run executed"
    assert mock_posthog_capture.call_args[1]["properties"]["add_isolated_node_eval"]


@pytest.mark.integration
@patch("posthog.capture")
def test_send_event_if_custom_error_raised(mock_posthog_capture):
    with pytest.raises(PipelineSchemaError):
        raise PipelineSchemaError
    # todo replace [1] with .kwargs when moving from python 3.7 to 3.8 in CI
    assert mock_posthog_capture.call_args[1]["event"] == "PipelineSchemaError raised"


def num_lines(path: Path):
    if path.is_file():
        with open(path, "r") as f:
            return len(f.readlines())
    return 0


@pytest.mark.integration
@patch("posthog.capture")
def test_write_to_file(mock_posthog_capture, monkeypatch):
    monkeypatch.setattr(telemetry, "LOG_PATH", Path("~/.haystack/telemetry_test.log").expanduser())
    num_lines_before = num_lines(telemetry.LOG_PATH)
    send_custom_event(event="test")
    num_lines_after = num_lines(telemetry.LOG_PATH)
    assert num_lines_before == num_lines_after

    enable_writing_events_to_file()
    num_lines_before = num_lines(telemetry.LOG_PATH)
    send_custom_event(event="test")
    num_lines_after = num_lines(telemetry.LOG_PATH)
    assert num_lines_before + 1 == num_lines_after

    disable_writing_events_to_file()
    num_lines_before = num_lines(telemetry.LOG_PATH)
    send_custom_event(event="test")
    num_lines_after = num_lines(telemetry.LOG_PATH)
    assert num_lines_before == num_lines_after
    _delete_telemetry_file(TelemetryFileType.LOG_FILE)


@pytest.mark.integration
@patch("posthog.capture")
def test_disable_enable_telemetry(mock_posthog_capture, monkeypatch):
    monkeypatch.setattr(telemetry, "HAYSTACK_TELEMETRY_ENABLED", "HAYSTACK_TELEMETRY_ENABLED_TEST")
    monkeypatch.setattr(telemetry, "CONFIG_PATH", Path("~/.haystack/config_test.yaml").expanduser())
    # config_test.yaml doesn't exist yet and won't be created automatically because the global user_id might have been set already by other tests
    _write_telemetry_config()
    send_custom_event(event="test")
    send_custom_event(event="test")
    assert mock_posthog_capture.call_count == 2, "two events should be sent"

    disable_telemetry()
    send_custom_event(event="test")
    assert mock_posthog_capture.call_count == 3, "one additional event should be sent"
    # todo replace [1] with .kwargs when moving from python 3.7 to 3.8 in CI
    assert mock_posthog_capture.call_args[1]["event"] == "telemetry disabled", "a final event should be sent"
    send_custom_event(event="test")
    assert mock_posthog_capture.call_count == 3, "no additional event should be sent"

    enable_telemetry()
    send_custom_event(event="test")
    assert mock_posthog_capture.call_count == 4, "one additional event should be sent"
