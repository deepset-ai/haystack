# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Optional


from canals.testing import BaseTestComponent
from canals.component import component


@component
class Threshold:
    """
    Redirects the value, unchanged, along a different connection whether the value is above
    or below the given threshold.

    Single input, double output decision component.

    :param threshold: the number to compare the input value against. This is also a parameter.
    """

    @component.input  # type: ignore
    def input(self):
        class Input:
            value: int
            threshold: int = 10

        return Input

    @component.output  # type: ignore
    def output(self):
        class Output:
            above: int
            below: int

        return Output

    def __init__(self, threshold: Optional[int] = None):
        """
        :param threshold: the number to compare the input value against.
        """
        if threshold:
            self.defaults = {"threshold": threshold}

    def run(self, data):
        if data.value < data.threshold:
            return self.output(above=None, below=data.value)
        return self.output(above=data.value, below=None)


class TestThreshold(BaseTestComponent):
    def test_saveload_default(self, tmp_path):
        self.assert_can_be_saved_and_loaded_in_pipeline(Threshold(), tmp_path)

    def test_saveload_threshold(self, tmp_path):
        self.assert_can_be_saved_and_loaded_in_pipeline(Threshold(threshold=3), tmp_path)

    def test_threshold(self):
        component = Threshold()

        results = component.run(component.input(value=5, threshold=10))
        assert results == component.output(above=None, below=5)

        results = component.run(component.input(value=15, threshold=10))
        assert results == component.output(above=15, below=None)
