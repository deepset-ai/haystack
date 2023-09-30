import pytest

from haystack.nodes.prompt.prompt_model import PromptModelInvocationLayer
from haystack.nodes.prompt.invocation_layer import HFLocalInvocationLayer, HFInferenceEndpointInvocationLayer


@pytest.mark.unit
def test_invocation_layer_order():
    """
    Checks that the huggingface invocation layer is positioned further down the list of providers
    as they can time out or be slow to respond.
    """
    invocation_layers = PromptModelInvocationLayer.invocation_layer_providers
    assert HFLocalInvocationLayer in invocation_layers
    assert HFInferenceEndpointInvocationLayer in invocation_layers
    index_hf = invocation_layers.index(HFLocalInvocationLayer) + 1
    index_hf_inference = invocation_layers.index(HFInferenceEndpointInvocationLayer) + 1
    assert index_hf > len(invocation_layers) / 2
    assert index_hf_inference > len(invocation_layers) / 2
