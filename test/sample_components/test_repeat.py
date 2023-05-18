# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import TypeVar, Any, List

from dataclasses import make_dataclass, dataclass

import pytest

from canals.testing import BaseTestComponent
from canals.component import component, ComponentInput, ComponentOutput


@component
class Repeat:
    """
    Repeats the input value on all outputs.
    """

    @dataclass
    class Input(ComponentInput):
        value: int

    def __init__(self, outputs: List[str] = ["output_1", "output_2", "output_3"]):
        self.outputs = outputs
        self._output_type = make_dataclass(
            "Output", fields=[(val, int, None) for val in outputs], bases=(ComponentOutput,)
        )

    @property
    def output_type(self):
        return self._output_type

    def run(self, data: Input):
        output_dataclass = self.output_type()
        for output in self.outputs:
            setattr(output_dataclass, output, data.value)
        return output_dataclass


class TestRepeat(BaseTestComponent):
    def test_saveload_default(self, tmp_path):
        self.assert_can_be_saved_and_loaded_in_pipeline(Repeat(), tmp_path)

    def test_saveload_outputs(self, tmp_path):
        self.assert_can_be_saved_and_loaded_in_pipeline(Repeat(outputs=["one", "two"]), tmp_path)

    def test_repeat_default(self):
        component = Repeat()
        results = component.run(Repeat.Input(value=10))
        assert results == component.output_type(output_1=10, output_2=10, output_3=10)
        assert component.init_parameters == {}

    def test_repeat_init(self):
        component = Repeat(outputs=["one", "two"])
        results = component.run(Repeat.Input(value=10))
        assert results == component.output_type(one=10, two=10)
        assert component.init_parameters == {"outputs": ["one", "two"]}
