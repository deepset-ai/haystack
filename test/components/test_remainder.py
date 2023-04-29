from dataclasses import dataclass, make_dataclass
import pytest

from functools import partialmethod
import inspect

from canals import component


@component
class Remainder:
    """
    Redirects the value, unchanged, along the connection corresponding to the remainder
    of a division. For example, if `divisor=3`, the value `5` would be sent along
    the second output connection.

    Single input, multi output decision component. Order of output connections is critical.
    """

    def __init__(self, divisor: int = 2):
        self.divisor = divisor
        self.output_type = make_dataclass("Output", [(f"remainder_is_{val}", int, None) for val in range(self.divisor)])

    def run(self, value: int):
        """
        :param divisor: the number to divide the input value for.
        :param input: the name of the input connection.
        :param outputs: the name of the output connections. Must be equal in length to the
            divisor (if dividing by 3, you must give exactly three output names).
            Ordering is important.
        """
        remainder = value % self.divisor
        output = self.output_type()
        setattr(output, f"remainder_is_{remainder}", value)
        return output


def test_remainder_default():
    component = Remainder()
    results = component.run(value=3)
    assert results == component.output_dataclass(remainder_is_1=3)
