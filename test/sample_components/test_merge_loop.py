# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from canals.testing import BaseTestComponent
from sample_components import MergeLoop


class TestMergeLoop(BaseTestComponent):
    # TODO
    # def test_saveload_builtin_type(self, tmp_path):
    #     self.assert_can_be_saved_and_loaded_in_pipeline(
    #         MergeLoop(expected_type=int, inputs=["in_1", "in_2"]), tmp_path
    #     )

    # def test_saveload_complex_type(self, tmp_path):
    #     self.assert_can_be_saved_and_loaded_in_pipeline(MergeLoop(expected_type=List[int]), tmp_path)

    # def test_saveload_object_type(self, tmp_path):
    #     class MyObject: ...
    #     self.assert_can_be_saved_and_loaded_in_pipeline(MergeLoop(expected_type=MyObject()), tmp_path)

    def test_merge_first(self):
        component = MergeLoop(expected_type=int, inputs=["in_1", "in_2"])
        results = component.run(in_1=5)
        assert results == {"value": 5}
        assert component.init_parameters == {"expected_type": "<class 'int'>", "inputs": ["in_1", "in_2"]}

    def test_merge_second(self):
        component = MergeLoop(expected_type=int, inputs=["in_1", "in_2"])
        results = component.run(in_2=5)
        assert results == {"value": 5}
        assert component.init_parameters == {"expected_type": "<class 'int'>", "inputs": ["in_1", "in_2"]}

    def test_merge_nones(self):
        component = MergeLoop(expected_type=int, inputs=["in_1", "in_2", "in_3"])
        results = component.run()
        assert results == {"value": None}
        assert component.init_parameters == {"expected_type": "<class 'int'>", "inputs": ["in_1", "in_2", "in_3"]}

    def test_merge_one(self):
        component = MergeLoop(expected_type=int, inputs=["in_1"])
        results = component.run(in_1=1)
        assert results == {"value": 1}
        assert component.init_parameters == {"expected_type": "<class 'int'>", "inputs": ["in_1"]}

    def test_merge_one_none(self):
        component = MergeLoop(expected_type=int, inputs=[])
        results = component.run()
        assert results == {"value": None}
        assert component.init_parameters == {"expected_type": "<class 'int'>", "inputs": []}
