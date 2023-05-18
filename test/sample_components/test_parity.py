# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Optional

from dataclasses import dataclass
import pytest

from canals.testing import BaseTestComponent
from canals.component import component, ComponentInput, ComponentOutput


@component
class Parity:
    """
    Redirects the value, unchanged, along the 'even' connection if even, or along the 'odd' one if odd.
    """

    @dataclass
    class Input(ComponentInput):
        value: int

    @dataclass
    class Output(ComponentOutput):
        even: Optional[int] = None
        odd: Optional[int] = None

    def run(self, data: Input) -> Output:
        """
        :param value: The value to check for parity
        """
        remainder = data.value % 2
        if remainder:
            return Parity.Output(odd=data.value)
        return Parity.Output(even=data.value)


class TestParity(BaseTestComponent):
    def test_saveload_default(self, tmp_path):
        self.assert_can_be_saved_and_loaded_in_pipeline(Parity(), tmp_path)

    def test_parity(self):
        component = Parity()
        results = component.run(Parity.Input(value=1))
        assert results == Parity.Output(odd=1)
        results = component.run(Parity.Input(value=2))
        assert results == Parity.Output(even=2)
