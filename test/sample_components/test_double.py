# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

import pytest

from canals.component import component, ComponentInput, ComponentOutput
from canals.testing import BaseTestComponent


@component
class Double:
    """
    Doubles the input value.
    """

    @dataclass
    class Input(ComponentInput):
        value: int

    @dataclass
    class Output(ComponentOutput):
        value: int

    def run(self, data: Input) -> Output:
        """
        Doubles the input value
        """
        return Double.Output(value=data.value * 2)


import pytest
from canals.testing import BaseTestComponent


class TestDouble(BaseTestComponent):
    @pytest.fixture
    def components(self):
        return [Double()]

    def test_double_default(self):
        component = Double()
        results = component.run(Double.Input(value=10))
        assert results == Double.Output(value=20)
        assert component.init_parameters == {}
