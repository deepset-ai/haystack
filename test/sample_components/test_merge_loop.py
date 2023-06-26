# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import builtins
from typing import List, Union, Optional
from dataclasses import make_dataclass, is_dataclass, asdict

from canals.component import component
from canals.testing import BaseTestComponent


@component
class MergeLoop:
    """
    Takes multiple inputs and returns the first one that is not None.
    """

    def __init__(self, expected_type: Union[type, str], inputs: List[str] = ["value_1", "value_2"]):
        if isinstance(expected_type, str):
            self.expected_type = getattr(builtins, expected_type)
        else:
            self.expected_type = expected_type
        self.init_parameters = {"expected_type": self.expected_type.__name__}
        # mypy complains that we can't Optional is not a type, so we ignore the error
        # cause we consider this to be correct
        self._input = make_dataclass("Input", fields=[(f, Optional[self.expected_type]) for f in inputs])  # type: ignore

    @component.input  # type: ignore
    def input(self):
        return self._input

    @component.output  # type: ignore
    def output(self):
        class Output:
            value: self.expected_type  # type: ignore

        return Output

    def run(self, data):
        """
        Takes some inputs and returns the first one that is not None.
        """
        values = []
        if is_dataclass(data):
            values = asdict(data).values()
        for v in values:
            if v is not None:
                return self.output(value=v)
        return self.output(value=None)


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
        component = MergeLoop(expected_type=int, inputs=["in_1", "in_2"])
        results = component.run(component.input(5, None))
        assert results == component.output(value=5)

    def test_merge_second(self):
        component = MergeLoop(expected_type=int, inputs=["in_1", "in_2"])
        results = component.run(component.input(None, 5))
        assert results == component.output(value=5)

    def test_merge_nones(self):
        component = MergeLoop(expected_type=int, inputs=["in_1", "in_2", "in_3"])
        results = component.run(component.input(None, None, None))
        assert results == component.output(value=None)

    def test_merge_one(self):
        component = MergeLoop(expected_type=int)
        results = component.run(component.input(1))
        assert results == component.output(value=1)

    def test_merge_one_none(self):
        component = MergeLoop(expected_type=int)
        results = component.run(component.input())
        assert results == component.output(value=None)
