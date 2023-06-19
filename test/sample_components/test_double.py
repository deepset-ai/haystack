# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from canals.component import component
from canals.testing import BaseTestComponent


@component
class Double:
    """
    Doubles the input value.
    """

    @component.input  # type: ignore
    def input(self):
        class Input:
            value: int

        return Input

    @component.output  # type: ignore
    def output(self):
        class Output:
            value: int

        return Output

    def run(self, data):
        """
        Doubles the input value
        """
        return self.output(value=data.value * 2)


class TestDouble(BaseTestComponent):
    def test_saveload_default(self, tmp_path):
        self.assert_can_be_saved_and_loaded_in_pipeline(Double(), tmp_path)

    def test_double_default(self):
        component = Double()
        results = component.run(component.input(value=10))
        assert results == component.output(value=20)
        assert component.init_parameters == {}
