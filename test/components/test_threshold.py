from typing import Optional

from dataclasses import dataclass

from canals import component


@component
class Threshold:
    """
    Redirects the value, unchanged, along a different connection whether the value is above
    or below the given threshold.

    Single input, double output decision component.

    :param threshold: the number to compare the input value against. This is also a parameter.
    """

    @dataclass
    class Output:
        above: int
        below: int

    def __init__(
        self,
        threshold: int,
    ):
        """
        :param threshold: the number to compare the input value against.
        """
        self.threshold = threshold

    def run(self, value: int, threshold: Optional[int] = None) -> Output:
        threshold = threshold if threshold is not None else self.threshold
        if value < threshold:
            return Threshold.Output(above=None, below=value)  # type: ignore
        return Threshold.Output(above=value, below=None)  # type: ignore


def test_below_default():
    component = Threshold(threshold=10)
    results = component.run(value=5)
    assert results == Threshold.Output(above=None, below=5)

    results = component.run(value=15)
    assert results == Threshold.Output(above=15, below=None)

    results = component.run(value=15, threshold=20)
    assert results == Threshold.Output(above=None, below=15)

    assert component.init_parameters == {"threshold": 10}
