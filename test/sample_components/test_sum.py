# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Optional
from dataclasses import make_dataclass, asdict, is_dataclass

from canals.testing import BaseTestComponent
from canals.component import component


@component
class Sum:
    """
    Sums the values of all the input connections together.
    """

    def __init__(self, inputs=["value_1"]) -> None:
        # mypy complains that we can't Optional is not a type, so we ignore the error
        # cause we consider this to be correct
        self._input = make_dataclass("Input", fields=[(f, Optional[int]) for f in inputs])  # type: ignore

    @component.input  # type: ignore
    def input(self):
        return self._input

    @component.output  # type: ignore
    def output(self):
        class Output:
            total: int

        return Output

    def run(self, data):
        values = []
        if is_dataclass(data):
            values = [n for n in asdict(data).values() if n]
        return self.output(total=sum(values))


class TestSum(BaseTestComponent):
    def test_saveload_default(self, tmp_path):
        self.assert_can_be_saved_and_loaded_in_pipeline(Sum(), tmp_path)

    def test_sum_no_values(self):
        component = Sum(inputs=[])
        results = component.run(component.input())
        assert results == component.output(total=0)
        assert component.init_parameters == {"inputs": []}

    def test_sum_one_value(self):
        component = Sum()
        results = component.run(component.input(10))
        assert results == component.output(total=10)
        assert component.init_parameters == {}

    def test_sum_few_values(self):
        component = Sum(inputs=["value_1", "value_2", "value_3"])
        results = component.run(component.input(10, 11, 12))
        assert results == component.output(total=33)
        assert component.init_parameters == {"inputs": ["value_1", "value_2", "value_3"]}
