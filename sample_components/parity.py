# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from canals import component


@component
class Parity:
    """
    Redirects the value, unchanged, along the 'even' connection if even, or along the 'odd' one if odd.
    """

    @component.input
    def input(self):
        class Input:
            value: int

        return Input

    @component.output
    def output(self):
        class Output:
            even: int = None
            odd: int = None

        return Output

    def run(self, data):
        """
        :param value: The value to check for parity
        """
        remainder = data.value % 2
        if remainder:
            return self.output(odd=data.value)
        return self.output(even=data.value)
