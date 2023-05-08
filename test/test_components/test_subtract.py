from dataclasses import dataclass

import pytest

from canals.testing import BaseTestComponent
from canals import component


@component
class Subtract:
    """
    Compute the difference between two values.
    """

    @dataclass
    class Output:
        difference: int

    def run(self, first_value: int, second_value: int) -> Output:
        """
        :param first_value: name of the connection carrying the value to subtract from.
        :param second_value: name of the connection carrying the value to subtract.
        """
        return Subtract.Output(difference=first_value - second_value)


class TestSubtract(BaseTestComponent):
    @pytest.fixture
    def components(self):
        return [Subtract()]

    def test_subtract(self):
        component = Subtract()
        results = component.run(first_value=10, second_value=7)
        assert results == Subtract.Output(difference=3)
        assert component._init_parameters == {}
