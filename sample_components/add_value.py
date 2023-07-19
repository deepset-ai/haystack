# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Optional

from canals import component


@component
class AddFixedValue:
    """
    Adds the value of `add` to `value`. If not given, `add` defaults to 1.
    """

    @component.input
    def input(self):
        class Input:
            value: int
            add: int

        return Input

    @component.output
    def output(self):
        class Output:
            value: int

        return Output

    def __init__(self, add: Optional[int] = 1):
        if add:
            self.defaults = {"add": add}

    def run(self, data):
        return self.output(value=data.value + data.add)
