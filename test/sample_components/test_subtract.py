# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

import pytest

from canals.testing import BaseTestComponent
from canals.component import component, ComponentInput, ComponentOutput


@component
class Subtract:
    """
    Compute the difference between two values.
    """

    @dataclass
    class Input(ComponentInput):
        first_value: int
        second_value: int

    @dataclass
    class Output(ComponentOutput):
        difference: int

    def run(self, data: Input) -> Output:
        """
        :param first_value: name of the connection carrying the value to subtract from.
        :param second_value: name of the connection carrying the value to subtract.
        """
        return Subtract.Output(difference=data.first_value - data.second_value)


class TestSubtract(BaseTestComponent):
    @pytest.fixture
    def components(self):
        return [Subtract()]

    def test_subtract(self):
        component = Subtract()
        results = component.run(Subtract.Input(first_value=10, second_value=7))
        assert results == Subtract.Output(difference=3)
        assert component._init_parameters == {}
