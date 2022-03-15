from time import sleep

from unittest.mock import patch

from haystack.telemetry import NonPrivateParameters, send_event


def test_private_params_not_tracked():
    NonPrivateParameters.param_names = ["top_k", "model_name_or_path"]
    params = {"hostname": "private_hostname", "top_k": 2}
    tracked_params = NonPrivateParameters.apply_filter(params)

    expected_params = {"top_k": 2}
    assert tracked_params == expected_params


def test_non_private_params_tracked():
    NonPrivateParameters.param_names = ["top_k", "model_name_or_path"]
    params = {"model_name_or_path": "test-model", "top_k": 2}
    non_private_params = NonPrivateParameters.apply_filter(params)
    assert non_private_params == params


def test_only_private_params():
    NonPrivateParameters.param_names = ["top_k", "model_name_or_path"]
    non_private_params = NonPrivateParameters.apply_filter({})
    assert non_private_params == {}

    NonPrivateParameters.param_names = []
    non_private_params = NonPrivateParameters.apply_filter({"param": "value"})
    assert non_private_params == {}


@patch("posthog.capture")
def test_send_event_via_decorator(mock_posthog_capture):
    class TestClass:
        @send_event
        def run(self, add_isolated_node_eval: bool = False):
            pass

    test_class = TestClass()
    test_class.run(add_isolated_node_eval=True)
    sleep(1)
    #todo replace [1] with .kwargs when moving from python 3.7 to 3.8 in CI
    assert mock_posthog_capture.call_args[1]['event'] == 'TestClass.run executed'
    assert mock_posthog_capture.call_args[1]['properties']['add_isolated_node_eval']
