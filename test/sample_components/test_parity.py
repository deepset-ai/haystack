# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Optional


from canals.testing import BaseTestComponent
from canals.component import component


@component
class Parity:
    """
    Redirects the value, unchanged, along the 'even' connection if even, or along the 'odd' one if odd.
    """

    @component.input  # type: ignore
    def input(self):
        class Input:
            value: int

        return Input

    @component.output  # type: ignore
    def output(self):
        class Output:
            even: int = None
            odd: int = None

        return Output

    def run(self, data):
        """
        :param value: The value to check for parity
        """
        remainder = data.value % 2
        if remainder:
            return self.output(odd=data.value)
        return self.output(even=data.value)


class TestParity(BaseTestComponent):
    def test_saveload_default(self, tmp_path):
        self.assert_can_be_saved_and_loaded_in_pipeline(Parity(), tmp_path)

    def test_parity(self):
        component = Parity()
        results = component.run(component.input(value=1))
        assert results == component.output(odd=1)
        results = component.run(component.input(value=2))
        assert results == component.output(even=2)
