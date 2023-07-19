# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Optional
from dataclasses import make_dataclass, asdict, is_dataclass

from canals import component


@component
class Sum:
    """
    Sums the values of all the input connections together.
    """

    def __init__(self, inputs=["value_1"]) -> None:
        # mypy complains that we can't Optional is not a type, so we ignore the error
        # cause we consider this to be correct
        self._input = make_dataclass("Input", fields=[(f, Optional[int]) for f in inputs])  # type: ignore

    @component.input
    def input(self):
        return self._input

    @component.output
    def output(self):
        class Output:
            total: int

        return Output

    def run(self, data):
        values = []
        if is_dataclass(data):
            values = [n for n in asdict(data).values() if n]
        return self.output(total=sum(values))
