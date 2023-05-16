# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import List

from dataclasses import dataclass

import pytest

from canals.component import component, ComponentInput, ComponentOutput
from canals.testing import BaseTestComponent


@component
class MergeLoop:
    """
    Takes two input components and returns the first one that is not None.

    In case both received a value, priority is given to 'first'.

    Always initialize this class by passing the expected type, like: `MergeLoop(expected_type=int)`.
    """

    def __init__(self, expected_type: type):
        self.expected_type = expected_type
        self._init_parameters = {"expected_type": expected_type.__name__}

    @property
    def input_type(self) -> ComponentInput:
        @dataclass
        class Input(ComponentInput):
            values: List[self.expected_type]

            def __init__(self, *values: self.expected_type):
                self.values = list(values)

        return Input

    @property
    def output_type(self) -> ComponentOutput:
        @dataclass
        class Output(ComponentOutput):
            value: self.expected_type

        return Output

    def run(self, data):
        """
        Takes some inputs and returns the first one that is not None.

        In case it receives more than one value, priority is given to the first.
        """
        for v in data.values:
            if v is not None:
                return self.output_type(value=v)
        return self.output_type(value=None)


class TestMergeLoop(BaseTestComponent):
    @pytest.fixture
    def components(self):

        # Serialization is broken :'(
        return [
            # MergeLoop(expected_type=int),
            # MergeLoop(expected_type=str),
            # MergeLoop(expected_type=List[str])
        ]

    def test_merge_first(self):
        component = MergeLoop(expected_type=int)
        results = component.run(component.input_type(5, None))
        assert results.__dict__ == component.output_type(value=5).__dict__

    def test_merge_second(self):
        component = MergeLoop(expected_type=int)
        results = component.run(component.input_type(None, 5))
        assert results.__dict__ == component.output_type(value=5).__dict__

    def test_merge_nones(self):
        component = MergeLoop(expected_type=int)
        results = component.run(component.input_type(None, None, None))
        assert results.__dict__ == component.output_type(value=None).__dict__

    def test_merge_one(self):
        component = MergeLoop(expected_type=int)
        results = component.run(component.input_type(1))
        assert results.__dict__ == component.output_type(value=1).__dict__

    def test_merge_one_none(self):
        component = MergeLoop(expected_type=int)
        results = component.run(component.input_type())
        assert results.__dict__ == component.output_type(value=None).__dict__
