# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any
from canals.component import component


def make_component(input=Any, output=Any):
    @component
    class Component:
        @component.input
        def input(self):
            class Input:
                value: input

            return Input

        @component.output
        def output(self):
            class Output:
                value: output

            return Output

        def run(self, data):
            return self.output()

    return Component()
