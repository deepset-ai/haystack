import pytest

from haystack import Pipeline
from haystack.core.errors import PipelineConnectError
from haystack.testing.sample_components import Double


def test_connect_component_twice():
    pipe = Pipeline()
    c1 = Double()
    c2 = Double()
    c3 = Double()
    pipe.add_component("c1", c1)
    pipe.add_component("c2", c2)
    pipe.add_component("c3", c3)
    pipe.connect("c1.value", "c2.value")
    # the following should be a no-op
    pipe.connect("c1.value", "c2.value")
    # this should fail instead
    with pytest.raises(PipelineConnectError):
        pipe.connect("c3.value", "c2.value")
