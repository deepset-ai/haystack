# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack.core.component import component
from haystack.core.component.types import Variadic


@component
class SelfLoop:
    """
    Decreases the initial value in steps of 1 until the target value is reached.

    For no good reason it uses a self-loop to do so :)
    """

    def __init__(self, target: int = 0):
        self.target = target

    @component.output_types(current_value=int, final_result=int)
    def run(self, values: Variadic[int]):
        """Decreases the input value in steps of 1 until the target value is reached."""
        value = values[0]  # type: ignore
        value -= 1
        if value == self.target:
            return {"final_result": value}
        return {"current_value": value}
