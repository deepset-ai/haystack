from time import sleep

from unittest.mock import patch, PropertyMock

from haystack.telemetry import NonPrivateParameters, send_event


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
