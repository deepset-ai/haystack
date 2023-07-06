import pytest

from haystack.nodes.prompt.prompt_model import PromptModelInvocationLayer
from haystack.nodes.prompt.invocation_layer import HFLocalInvocationLayer, HFInferenceEndpointInvocationLayer


@pytest.mark.unit
def test_invocation_layer_order():
    """
    Checks that the huggingface invocation layer is checked late because it can timeout/be slow to respond.
    """
    last_invocation_layers = set(PromptModelInvocationLayer.invocation_layer_providers[-5:])
    assert HFLocalInvocationLayer in last_invocation_layers
    assert HFInferenceEndpointInvocationLayer in last_invocation_layers
