# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from canals import component


@component
class Double:
    """
    Doubles the input value.
    """

    @component.input
    def input(self):
        class Input:
            value: int

        return Input

    @component.output
    def output(self):
        class Output:
            value: int

        return Output

    def run(self, data):
        """
        Doubles the input value
        """
        return self.output(value=data.value * 2)
