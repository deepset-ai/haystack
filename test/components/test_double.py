from dataclasses import dataclass
from canals import component, component_input, component_output


@component
class Double:
    """
    Doubles the value in input.

    Single input single output component. Doesn't take parameters.
    """

    @component_input
    class Input:
        value: int

    @component_output
    class Output:
        value: int

    def run(self, data: Input) -> Output:
        """
        Doubles the input value
        """
        return Double.Output(value=data.value * 2)  # type: ignore


def test_double_default():
    component = Double()
    results = component.run(Double(value=10))
    assert results == Double.Output(value=20)
    assert component.init_parameters == {}
