from dataclasses import dataclass
from canals import component, component_input, component_output


@component
class AddFixedValue:
    """
    Adds the value of `add` to `value`. If not given, `add` defaults to 1.
    """

    @component_input
    class Input:
        value: int
        add: int = 1

    @component_output
    class Output:
        value: int

    def run(self, data: Input) -> Output:
        return AddFixedValue.Output(value=data.value + data.add)  # type: ignore


def test_addvalue_no_add():
    component = AddFixedValue()
    results = component.run(AddFixedValue.Input(value=10))
    assert results == AddFixedValue.Output(value=11)
    assert component.init_parameters == {}


def test_addvalue_with_add():
    component = AddFixedValue()
    results = component.run(AddFixedValue.Input(value=50, add=10))
    assert results == AddFixedValue.Output(value=60)
    assert component.init_parameters == {}
