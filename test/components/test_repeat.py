from typing import TypeVar, Any

from dataclasses import dataclass
from canals import component


@component
class Repeat:
    """
    Repeats the input value on all outputs.

    CAN REPEAT ONLY INTS.
    """

    @dataclass
    class Output:
        first: int
        second: int

    def run(self, value: int) -> Output:
        return Repeat.Output(first=value, second=value)


def test_repeat_default():
    component = Repeat()
    results = component.run(value=10)
    assert results == Repeat.Output(first=10, second=10)
    assert component.init_parameters == {}
