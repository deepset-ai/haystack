# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Optional

from canals import component


@component
class Threshold:
    """
    Redirects the value, unchanged, along a different connection whether the value is above
    or below the given threshold.

    Single input, double output decision component.

    :param threshold: the number to compare the input value against. This is also a parameter.
    """

    @component.input
    def input(self):
        class Input:
            value: int
            threshold: int = 10

        return Input

    @component.output
    def output(self):
        class Output:
            above: int
            below: int

        return Output

    def __init__(self, threshold: Optional[int] = None):
        """
        :param threshold: the number to compare the input value against.
        """
        if threshold:
            self.defaults = {"threshold": threshold}

    def run(self, data):
        if data.value < data.threshold:
            return self.output(above=None, below=data.value)
        return self.output(above=data.value, below=None)
