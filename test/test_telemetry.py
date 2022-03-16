from pathlib import Path
from time import sleep

from unittest.mock import patch, PropertyMock

import haystack
from haystack.telemetry import NonPrivateParameters, send_event, enable_writing_events_to_file, \
    disable_writing_events_to_file, send_custom_event, _delete_telemetry_log_file, disable_telemetry, enable_telemetry


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
    sleep(1)
    # todo replace [1] with .kwargs when moving from python 3.7 to 3.8 in CI
    assert mock_posthog_capture.call_args[1]["event"] == "TestClass.run executed"
    assert mock_posthog_capture.call_args[1]["properties"]["add_isolated_node_eval"]


def num_lines(path: Path):
    if path.is_file():
        with open(path, 'r') as f:
            return len(f.readlines())
    return 0


@patch("haystack.telemetry.LOG_PATH", Path("~/.haystack/telemetry_test.log").expanduser())
def test_write_to_file():
    num_lines_before = num_lines(haystack.telemetry.LOG_PATH)
    send_custom_event(event="test")
    sleep(1)
    num_lines_after = num_lines(haystack.telemetry.LOG_PATH)
    assert num_lines_before == num_lines_after

    enable_writing_events_to_file()
    num_lines_before = num_lines(haystack.telemetry.LOG_PATH)
    send_custom_event(event="test")
    sleep(1)
    num_lines_after = num_lines(haystack.telemetry.LOG_PATH)
    assert num_lines_before+1 == num_lines_after

    disable_writing_events_to_file()
    num_lines_before = num_lines(haystack.telemetry.LOG_PATH)
    send_custom_event(event="test")
    sleep(1)
    num_lines_after = num_lines(haystack.telemetry.LOG_PATH)
    assert num_lines_before == num_lines_after
    _delete_telemetry_log_file()


@patch("posthog.capture")
def test_disable_enable_telemetry(mock_posthog_capture):
    send_custom_event(event="test")
    sleep(1)
    assert mock_posthog_capture.call_count == 1, 'a single event should be sent'

    disable_telemetry()
    send_custom_event(event="test")
    sleep(1)
    assert mock_posthog_capture.call_count == 2, 'one additional final event should be sent'
    send_custom_event(event="test")
    sleep(1)
    assert mock_posthog_capture.call_count == 2, 'no additional event should be sent'

    enable_telemetry()
    send_custom_event(event="test")
    sleep(1)
    assert mock_posthog_capture.call_count == 3, 'one additional event should be sent'
