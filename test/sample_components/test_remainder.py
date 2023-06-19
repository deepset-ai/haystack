# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from dataclasses import make_dataclass

import pytest

from canals.testing import BaseTestComponent
from canals.component import component


@component
class Remainder:
    """
    Redirects the value, unchanged, along the connection corresponding to the remainder
    of a division. For example, if `divisor=3`, the value `5` would be sent along
    the second output connection.
    """

    @component.input  # type: ignore
    def input(self):
        class Input:
            value: int

        return Input

    def __init__(self, divisor: int = 2):
        if divisor == 0:
            raise ValueError("Can't divide by zero")
        self.divisor = divisor

        self._output_type = make_dataclass(
            "Output", fields=[(f"remainder_is_{val}", int, None) for val in range(divisor)]
        )

    @component.output  # type: ignore
    def output(self):
        return self._output_type

    def run(self, data):
        """
        :param value: the value to check the remainder of.
        """
        remainder = data.value % self.divisor
        output = self.output()
        setattr(output, f"remainder_is_{remainder}", data.value)
        return output


class TestRemainder(BaseTestComponent):
    def test_saveload_default(self, tmp_path):
        self.assert_can_be_saved_and_loaded_in_pipeline(Remainder(), tmp_path)

    def test_saveload_divisor(self, tmp_path):
        self.assert_can_be_saved_and_loaded_in_pipeline(Remainder(divisor=1), tmp_path)

    def test_remainder_default(self):
        component = Remainder()
        results = component.run(component.input(value=3))
        assert results == component.output(remainder_is_1=3)

    def test_remainder_with_divisor(self):
        component = Remainder(divisor=4)
        results = component.run(component.input(value=3))
        assert results == component.output(remainder_is_3=3)

    def test_remainder_zero(self):
        with pytest.raises(ValueError):
            Remainder(divisor=0)
