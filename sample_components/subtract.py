# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from canals import component


@component
class Subtract:
    """
    Compute the difference between two values.
    """

    @component.input
    def input(self):
        class Input:
            first_value: int
            second_value: int

        return Input

    @component.output
    def output(self):
        class Output:
            difference: int

        return Output

    def run(self, data):
        """
        :param first_value: name of the connection carrying the value to subtract from.
        :param second_value: name of the connection carrying the value to subtract.
        """
        return self.output(difference=data.first_value - data.second_value)
