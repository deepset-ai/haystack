from unittest.mock import Mock


def create_mock_layer_that_supports(model_name, response=["fake_response"]):
    """
    Create a mock invocation layer that supports the model_name and returns response.
    """

    def mock_supports(model_name_or_path, **kwargs):
        return model_name_or_path == model_name

    return Mock(**{"model_name_or_path": model_name, "supports": mock_supports, "invoke.return_value": response})
