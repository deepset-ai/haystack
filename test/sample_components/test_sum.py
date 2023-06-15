# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import List


from canals.testing import BaseTestComponent
from canals.component import component


@component
class Sum:
    """
    Sums the values of all the input connections together.
    """

    @component.input(variadic=True)
    def input(self):
        class Input:
            values: List[int]

        return Input

    @component.output
    def output(self):
        class Output:
            total: int

        return Output

    def run(self, data):
        return self.output(total=sum(data.values))


class TestSum(BaseTestComponent):
    def test_saveload_default(self, tmp_path):
        self.assert_can_be_saved_and_loaded_in_pipeline(Sum(), tmp_path)

    def test_sum_no_values(self):
        component = Sum()
        results = component.run(component.input())
        assert results == component.output(total=0)
        assert component.init_parameters == {}

    def test_sum_one_value(self):
        component = Sum()
        results = component.run(component.input(10))
        assert results == component.output(total=10)
        assert component.init_parameters == {}

    def test_sum_few_values(self):
        component = Sum()
        results = component.run(component.input(10, 11, 12))
        assert results == component.output(total=33)
        assert component.init_parameters == {}
