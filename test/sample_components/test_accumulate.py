# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from canals.testing import BaseTestComponent
from sample_components import Accumulate


def my_subtract(first, second):
    return first - second


class TestAccumulate(BaseTestComponent):
    def test_saveload_default(self, tmp_path):
        self.assert_can_be_saved_and_loaded_in_pipeline(Accumulate(), tmp_path)

    def test_saveload_function_as_string(self, tmp_path):
        self.assert_can_be_saved_and_loaded_in_pipeline(
            Accumulate(function="test.sample_components.test_accumulate.my_subtract"), tmp_path
        )

    def test_saveload_function_as_callable(self, tmp_path):
        self.assert_can_be_saved_and_loaded_in_pipeline(Accumulate(function=my_subtract), tmp_path)

    def test_accumulate_default(self):
        component = Accumulate()
        results = component.run(value=10)
        assert results == {"value": 10}
        assert component.state == 10

        results = component.run(value=1)
        assert results == {"value": 11}
        assert component.state == 11

        assert component.init_parameters == {}

    def test_accumulate_callable(self):
        component = Accumulate(function=my_subtract)

        results = component.run(value=10)
        assert results == {"value": -10}
        assert component.state == -10

        results = component.run(value=1)
        assert results == {"value": -11}
        assert component.state == -11

        assert component.init_parameters == {
            "function": "test.sample_components.test_accumulate.my_subtract",
        }

    def test_accumulate_string(self):
        component = Accumulate(function="test.sample_components.test_accumulate.my_subtract")

        results = component.run(value=10)
        assert results == {"value": -10}
        assert component.state == -10

        results = component.run(value=1)
        assert results == {"value": -11}
        assert component.state == -11

        assert component.init_parameters == {
            "function": "test.sample_components.test_accumulate.my_subtract",
        }
