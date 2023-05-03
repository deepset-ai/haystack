from typing import Optional  # TypedDict
from dataclasses import dataclass
from canals import component  # , component_input, component_output


@component
class AddFixedValue:
    """
    Adds the value of `add` to `value`. If not given, `add` defaults to 1.
    """

    @dataclass
    class Input:
        value: int
        add: int

    @dataclass
    class Output:
        value: int

    def __init__(self, defaults: Optional[Input] = None):
        self.defaults = defaults

    def run(self, data: Input) -> Output:
        return AddFixedValue.Output(value=data.value + data.add)


def test_addvalue_no_add():
    comp = AddFixedValue(defaults={"value": "hello"})
    results = comp.run(AddFixedValue.Input(value=10))
    assert results == AddFixedValue.Output(value=11)
    assert comp.init_parameters == {}


def test_addvalue_with_add():
    comp = AddFixedValue()
    results = comp.run(AddFixedValue.Input(value=50, add=10))
    assert results == AddFixedValue.Output(value=60)
    assert comp.init_parameters == {}
