# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from dataclasses import make_dataclass

from canals import component


@component
class Remainder:
    """
    Redirects the value, unchanged, along the connection corresponding to the remainder
    of a division. For example, if `divisor=3`, the value `5` would be sent along
    the second output connection.
    """

    @component.input
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

    @component.output
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
