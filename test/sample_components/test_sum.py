# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from canals.testing import BaseTestComponent
from sample_components import Sum


class TestSum(BaseTestComponent):
    def test_saveload_default(self, tmp_path):
        self.assert_can_be_saved_and_loaded_in_pipeline(Sum(inputs=["value_1", "value_1"]), tmp_path)

    def test_sum_expects_no_values_receives_no_values(self):
        component = Sum(inputs=[])
        results = component.run()
        assert results == {"total": 0}
        assert component.init_parameters == {"inputs": []}

    def test_sum_expects_no_values_receives_one_value(self):
        component = Sum(inputs=[])
        assert component.run(value_1=10) == {"total": 10}

    def test_sum_expects_one_value_receives_one_value(self):
        component = Sum(inputs=["value_1"])
        results = component.run(value_1=10)
        assert results == {"total": 10}
        assert component.init_parameters == {"inputs": ["value_1"]}

    def test_sum_expects_one_value_receives_wrong_value(self):
        component = Sum(inputs=["value_1"])
        assert component.run(something_else=10) == {"total": 10}

    def test_sum_expects_one_value_receives_few_values(self):
        component = Sum(inputs=["value_1"])
        assert component.run(value_1=10, value_2=2) == {"total": 12}

    def test_sum_expects_few_values_receives_right_values(self):
        component = Sum(inputs=["value_1", "value_2", "value_3"])
        results = component.run(value_1=10, value_2=11, value_3=12)
        assert results == {"total": 33}
        assert component.init_parameters == {"inputs": ["value_1", "value_2", "value_3"]}

    def test_sum_expects_few_values_receives_some_wrong_values(self):
        component = Sum(inputs=["value_1", "value_2", "value_3"])
        assert component.run(value_1=10, value_4=11, value_3=12) == {"total": 33}
