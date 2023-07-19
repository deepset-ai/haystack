# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from canals.testing import BaseTestComponent
from sample_components import MergeLoop


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
