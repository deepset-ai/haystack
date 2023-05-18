# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import List, Union
import builtins

from dataclasses import dataclass

from canals.component import component, VariadicComponentInput, ComponentOutput
from canals.testing import BaseTestComponent


@component
class MergeLoop:
    """
    Takes two input components and returns the first one that is not None.
    In case both received a value, priority is given to 'first'.
    """

    def __init__(self, expected_type: Union[type, str]):
        if isinstance(expected_type, str):
            type_ = getattr(builtins, expected_type)
        else:
            type_ = expected_type

        self.expected_type = type_
        self.init_parameters = {"expected_type": type_.__name__}

    @property
    def input_type(self):
        @dataclass
        class Input(VariadicComponentInput):
            values: List[self.expected_type]  # type: ignore

        return Input

    @property
    def output_type(self):
        @dataclass
        class Output(ComponentOutput):
            value: self.expected_type  # type: ignore

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
    def test_saveload_builtin_type(self, tmp_path):
        self.assert_can_be_saved_and_loaded_in_pipeline(MergeLoop(expected_type=int), tmp_path)

    # TODO
    # def test_saveload_complex_type(self, tmp_path):
    #     self.assert_can_be_saved_and_loaded_in_pipeline(MergeLoop(expected_type=List[int]), tmp_path)

    # def test_saveload_object_type(self, tmp_path):
    #     class MyObject: ...
    #     self.assert_can_be_saved_and_loaded_in_pipeline(MergeLoop(expected_type=MyObject()), tmp_path)

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
