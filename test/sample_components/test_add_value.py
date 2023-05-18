# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Optional

from dataclasses import dataclass

import pytest

from canals.component import component, ComponentInput, ComponentOutput
from canals.testing.test_component import BaseTestComponent


@component
class AddFixedValue:
    """
    Adds the value of `add` to `value`. If not given, `add` defaults to 1.
    """

    @dataclass
    class Input(ComponentInput):
        value: int
        add: int

    @dataclass
    class Output(ComponentOutput):
        value: int

    def __init__(self, add: Optional[int] = 1):
        if add:
            self.defaults = {"add": add}

    def run(self, data: Input) -> Output:
        return AddFixedValue.Output(value=data.value + data.add)


class TestAddFixedValue(BaseTestComponent):
    def test_saveload_default(self, tmp_path):
        self.assert_can_be_saved_and_loaded_in_pipeline(AddFixedValue(), tmp_path)

    def test_saveload_add(self, tmp_path):
        self.assert_can_be_saved_and_loaded_in_pipeline(AddFixedValue(add=2), tmp_path)

    def test_addvalue(self):
        component = AddFixedValue()
        results = component.run(AddFixedValue.Input(value=50, add=10))
        assert results == AddFixedValue.Output(value=60)
        assert component.init_parameters == {}
