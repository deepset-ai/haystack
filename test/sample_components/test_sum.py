# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import List

from dataclasses import dataclass

import pytest

from canals.testing import BaseTestComponent
from canals.component import component, VariadicComponentInput, ComponentOutput


@component
class Sum:
    """
    Sums the values of all the input connections together.
    """

    @dataclass
    class Input(VariadicComponentInput):
        values: List[int]

    @dataclass
    class Output(ComponentOutput):
        total: int

    def run(self, data: Input) -> Output:
        return Sum.Output(total=sum(data.values))


class TestSum(BaseTestComponent):
    def test_saveload_default(self, tmp_path):
        self.assert_can_be_saved_and_loaded_in_pipeline(Sum(), tmp_path)

    def test_sum_no_values(self):
        component = Sum()
        results = component.run(Sum.Input())
        assert results == Sum.Output(total=0)
        assert component.init_parameters == {}

    def test_sum_one_value(self):
        component = Sum()
        results = component.run(Sum.Input(10))
        assert results == Sum.Output(total=10)
        assert component.init_parameters == {}

    def test_sum_few_values(self):
        component = Sum()
        results = component.run(Sum.Input(10, 11, 12))
        assert results == Sum.Output(total=33)
        assert component.init_parameters == {}
