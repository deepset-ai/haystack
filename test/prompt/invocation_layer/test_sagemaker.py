import pytest

from haystack.nodes.prompt.invocation_layer import SageMakerInvocationLayer


@pytest.mark.integration
def test_supports():
    """
    Test that supports returns True for valid SageMakerInvocationLayer
    """

    assert SageMakerInvocationLayer.supports(
        model_name_or_path="jumpstart-example-tiiuae-falcon-40b-ins-2023-06-16-09-15-35-027",
        profile_name="Haystack-OSS-test",
    )


@pytest.mark.integration
def test_supports_not():
    """
    Test that supports returns False for invalid SageMakerInvocationLayer
    """
    assert not SageMakerInvocationLayer.supports("google/flan-t5-xxl", profile_name="Haystack-OSS-test")
    assert not SageMakerInvocationLayer.supports(
        model_name_or_path="jumpstart-example-tiiuae-falcon-40b-ins-2023-06-16-09-15-35-027"
    )
    assert not SageMakerInvocationLayer.supports(
        model_name_or_path="invalid-model-name", profile_name="invalid-profile"
    )
