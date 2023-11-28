# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Dict

import pytest

from haystack.core.errors import DeserializationError

from haystack.testing.sample_components import MergeLoop


def test_to_dict():
    component = MergeLoop(expected_type=int, inputs=["first", "second"])
    res = component.to_dict()
    assert res == {
        "type": "haystack.testing.sample_components.merge_loop.MergeLoop",
        "init_parameters": {"expected_type": "builtins.int", "inputs": ["first", "second"]},
    }


def test_to_dict_with_typing_class():
    component = MergeLoop(expected_type=Dict, inputs=["first", "second"])
    res = component.to_dict()
    assert res == {
        "type": "haystack.testing.sample_components.merge_loop.MergeLoop",
        "init_parameters": {"expected_type": "typing.Dict", "inputs": ["first", "second"]},
    }


def test_to_dict_with_custom_class():
    component = MergeLoop(expected_type=MergeLoop, inputs=["first", "second"])
    res = component.to_dict()
    assert res == {
        "type": "haystack.testing.sample_components.merge_loop.MergeLoop",
        "init_parameters": {
            "expected_type": "haystack.testing.sample_components.merge_loop.MergeLoop",
            "inputs": ["first", "second"],
        },
    }


def test_from_dict():
    data = {
        "type": "haystack.testing.sample_components.merge_loop.MergeLoop",
        "init_parameters": {"expected_type": "builtins.int", "inputs": ["first", "second"]},
    }
    component = MergeLoop.from_dict(data)
    assert component.expected_type == "builtins.int"
    assert component.inputs == ["first", "second"]


def test_from_dict_with_typing_class():
    data = {
        "type": "haystack.testing.sample_components.merge_loop.MergeLoop",
        "init_parameters": {"expected_type": "typing.Dict", "inputs": ["first", "second"]},
    }
    component = MergeLoop.from_dict(data)
    assert component.expected_type == "typing.Dict"
    assert component.inputs == ["first", "second"]


def test_from_dict_with_custom_class():
    data = {
        "type": "haystack.testing.sample_components.merge_loop.MergeLoop",
        "init_parameters": {"expected_type": "sample_components.merge_loop.MergeLoop", "inputs": ["first", "second"]},
    }
    component = MergeLoop.from_dict(data)
    assert component.expected_type == "haystack.testing.sample_components.merge_loop.MergeLoop"
    assert component.inputs == ["first", "second"]


def test_from_dict_without_expected_type():
    data = {
        "type": "haystack.testing.sample_components.merge_loop.MergeLoop",
        "init_parameters": {"inputs": ["first", "second"]},
    }
    with pytest.raises(DeserializationError) as exc:
        MergeLoop.from_dict(data)

    exc.match("Missing 'expected_type' field in 'init_parameters'")


def test_from_dict_without_inputs():
    data = {
        "type": "haystack.testing.sample_components.merge_loop.MergeLoop",
        "init_parameters": {"expected_type": "sample_components.merge_loop.MergeLoop"},
    }
    with pytest.raises(DeserializationError) as exc:
        MergeLoop.from_dict(data)

    exc.match("Missing 'inputs' field in 'init_parameters'")


def test_merge_first():
    component = MergeLoop(expected_type=int, inputs=["in_1", "in_2"])
    results = component.run(in_1=5)
    assert results == {"value": 5}


def test_merge_second():
    component = MergeLoop(expected_type=int, inputs=["in_1", "in_2"])
    results = component.run(in_2=5)
    assert results == {"value": 5}


def test_merge_nones():
    component = MergeLoop(expected_type=int, inputs=["in_1", "in_2", "in_3"])
    results = component.run()
    assert results == {}


def test_merge_one():
    component = MergeLoop(expected_type=int, inputs=["in_1"])
    results = component.run(in_1=1)
    assert results == {"value": 1}


def test_merge_one_none():
    component = MergeLoop(expected_type=int, inputs=[])
    results = component.run()
    assert results == {}
