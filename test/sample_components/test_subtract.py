# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from canals.testing import BaseTestComponent
from canals.component import component


@component
class Subtract:
    """
    Compute the difference between two values.
    """

    @component.input  # type: ignore
    def input(self):
        class Input:
            first_value: int
            second_value: int

        return Input

    @component.output  # type: ignore
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


class TestSubtract(BaseTestComponent):
    def test_saveload_default(self, tmp_path):
        self.assert_can_be_saved_and_loaded_in_pipeline(Subtract(), tmp_path)

    def test_subtract(self):
        component = Subtract()
        results = component.run(component.input(first_value=10, second_value=7))
        assert results == component.output(difference=3)
        assert component.init_parameters == {}
