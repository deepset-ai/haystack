from typing import Optional

from dataclasses import dataclass
import pytest

from canals.testing import BaseTestComponent
from canals import component


@component
class Parity:
    """
    Redirects the value, unchanged, along the 'even' connection if even, or along the 'odd' one if odd.
    """

    @dataclass
    class Output:
        even: int  # Can't use Optional or connections will fail (can't send a 'Optional[int]' into a strict 'int')
        odd: int

    def run(self, value: int) -> Output:
        """
        :param value: The value to check for parity
        """
        remainder = value % 2
        if remainder:
            return Parity.Output(even=None, odd=value)  # type: ignore  # (mypy doesn't like the missing Optional)
        return Parity.Output(even=value, odd=None)  # type: ignore  # (mypy doesn't like the missing Optional)


class TestParity(BaseTestComponent):
    @pytest.fixture
    def components(self):
        return [Parity()]

    def test_parity(self):
        component = Parity()
        results = component.run(value=1)
        assert results == Parity.Output(even=None, odd=1)  # type: ignore  #(mypy doesn't like the missing Optional)
        results = component.run(value=2)
        assert results == Parity.Output(even=2, odd=None)  # type: ignore  #(mypy doesn't like the missing Optional)
