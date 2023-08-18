# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Dict

import pytest

from canals.errors import DeserializationError
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

    def test_to_dict(self):
        component = MergeLoop(expected_type=int, inputs=["first", "second"])
        res = component.to_dict()
        assert res == {
            "hash": id(component),
            "type": "MergeLoop",
            "init_parameters": {"expected_type": "builtins.int", "inputs": ["first", "second"]},
        }

    def test_to_dict_with_typing_class(self):
        component = MergeLoop(expected_type=Dict, inputs=["first", "second"])
        res = component.to_dict()
        assert res == {
            "hash": id(component),
            "type": "MergeLoop",
            "init_parameters": {
                "expected_type": "typing.Dict",
                "inputs": ["first", "second"],
            },
        }

    def test_to_dict_with_custom_class(self):
        component = MergeLoop(expected_type=MergeLoop, inputs=["first", "second"])
        res = component.to_dict()
        assert res == {
            "hash": id(component),
            "type": "MergeLoop",
            "init_parameters": {
                "expected_type": "sample_components.merge_loop.MergeLoop",
                "inputs": ["first", "second"],
            },
        }

    def test_from_dict(self):
        data = {
            "hash": 12345,
            "type": "MergeLoop",
            "init_parameters": {"expected_type": "builtins.int", "inputs": ["first", "second"]},
        }
        component = MergeLoop.from_dict(data)
        assert component.expected_type == "builtins.int"
        assert component.inputs == ["first", "second"]

    def test_from_dict_with_typing_class(self):
        data = {
            "hash": 1234,
            "type": "MergeLoop",
            "init_parameters": {
                "expected_type": "typing.Dict",
                "inputs": ["first", "second"],
            },
        }
        component = MergeLoop.from_dict(data)
        assert component.expected_type == "typing.Dict"
        assert component.inputs == ["first", "second"]

    def test_from_dict_with_custom_class(self):
        data = {
            "hash": 12345,
            "type": "MergeLoop",
            "init_parameters": {
                "expected_type": "sample_components.merge_loop.MergeLoop",
                "inputs": ["first", "second"],
            },
        }
        component = MergeLoop.from_dict(data)
        assert component.expected_type == "sample_components.merge_loop.MergeLoop"
        assert component.inputs == ["first", "second"]

    def test_from_dict_without_expected_type(self):
        data = {
            "hash": 12345,
            "type": "MergeLoop",
            "init_parameters": {
                "inputs": ["first", "second"],
            },
        }
        with pytest.raises(DeserializationError) as exc:
            MergeLoop.from_dict(data)

        exc.match("Missing 'expected_type' field in 'init_parameters'")

    def test_from_dict_without_inputs(self):
        data = {
            "hash": 12345,
            "type": "MergeLoop",
            "init_parameters": {
                "expected_type": "sample_components.merge_loop.MergeLoop",
            },
        }
        with pytest.raises(DeserializationError) as exc:
            MergeLoop.from_dict(data)

        exc.match("Missing 'inputs' field in 'init_parameters'")

    def test_merge_first(self):
        component = MergeLoop(expected_type=int, inputs=["in_1", "in_2"])
        results = component.run(in_1=5)
        assert results == {"value": 5}

    def test_merge_second(self):
        component = MergeLoop(expected_type=int, inputs=["in_1", "in_2"])
        results = component.run(in_2=5)
        assert results == {"value": 5}

    def test_merge_nones(self):
        component = MergeLoop(expected_type=int, inputs=["in_1", "in_2", "in_3"])
        results = component.run()
        assert results == {"value": None}

    def test_merge_one(self):
        component = MergeLoop(expected_type=int, inputs=["in_1"])
        results = component.run(in_1=1)
        assert results == {"value": 1}

    def test_merge_one_none(self):
        component = MergeLoop(expected_type=int, inputs=[])
        results = component.run()
        assert results == {"value": None}
