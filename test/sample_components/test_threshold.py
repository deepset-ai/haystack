# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Optional

from dataclasses import dataclass

import pytest

from canals.testing import BaseTestComponent
from canals.component import component, ComponentInput, ComponentOutput


@component
class Threshold:
    """
    Redirects the value, unchanged, along a different connection whether the value is above
    or below the given threshold.

    Single input, double output decision component.

    :param threshold: the number to compare the input value against. This is also a parameter.
    """

    @dataclass
    class Input(ComponentInput):
        value: int
        threshold: int = 10

    @dataclass
    class Output(ComponentOutput):
        above: int
        below: int

    def __init__(self, threshold: Optional[int] = None):
        """
        :param threshold: the number to compare the input value against.
        """
        if threshold:
            self.defaults = {"threshold": threshold}

    def run(self, data: Input) -> Output:
        if data.value < data.threshold:
            return Threshold.Output(above=None, below=data.value)  # type: ignore
        return Threshold.Output(above=data.value, below=None)  # type: ignore


class TestThreshold(BaseTestComponent):
    def test_saveload_default(self, tmp_path):
        self.assert_can_be_saved_and_loaded_in_pipeline(Threshold(), tmp_path)

    def test_saveload_threshold(self, tmp_path):
        self.assert_can_be_saved_and_loaded_in_pipeline(Threshold(threshold=3), tmp_path)

    def test_threshold(self):
        component = Threshold()

        results = component.run(Threshold.Input(value=5, threshold=10))
        assert results == Threshold.Output(above=None, below=5)

        results = component.run(Threshold.Input(value=15, threshold=10))
        assert results == Threshold.Output(above=15, below=None)
