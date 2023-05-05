from typing import TypeVar, Any, List

from dataclasses import make_dataclass
from canals import component


@component
class Repeat:
    """
    Repeats the input value on all outputs.
    """

    def __init__(self, outputs: List[str] = ["output_1", "output_2", "output_3"]):
        self.outputs = outputs
        self._output_type = make_dataclass("Output", [(val, int, None) for val in outputs])

    @property
    def output_type(self):
        return self._output_type

    def run(self, value: int):
        output_dataclass = self.output_type()
        for output in self.outputs:
            setattr(output_dataclass, output, value)
        return output_dataclass


def test_repeat_default():
    component = Repeat()
    results = component.run(value=10)
    assert results == component.output_type(output_1=10, output_2=10, output_3=10)
    assert component._init_parameters == {}


def test_repeat_init():
    component = Repeat(outputs=["one", "two"])
    results = component.run(value=10)
    assert results == component.output_type(one=10, two=10)
    assert component._init_parameters == {"outputs": ["one", "two"]}
