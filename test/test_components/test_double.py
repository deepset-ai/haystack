from dataclasses import dataclass
from canals import component


@component
class Double:
    """
    Doubles the value in input.

    Single input single output component. Doesn't take parameters.
    """

    @dataclass
    class Output:
        value: int

    def run(self, value: int) -> Output:
        """
        Doubles the input value
        """
        return Double.Output(value=value * 2)


def test_double_default():
    component = Double()
    results = component.run(value=10)
    assert results == Double.Output(value=20)
    assert component._init_parameters == {}
