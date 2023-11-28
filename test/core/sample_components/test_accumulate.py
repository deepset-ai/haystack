# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from haystack.testing.sample_components.accumulate import Accumulate, _default_function


def my_subtract(first, second):
    return first - second


def test_to_dict():
    accumulate = Accumulate()
    res = accumulate.to_dict()
    assert res == {
        "type": "haystack.testing.sample_components.accumulate.Accumulate",
        "init_parameters": {"function": "haystack.testing.sample_components.accumulate._default_function"},
    }


def test_to_dict_with_custom_function():
    accumulate = Accumulate(function=my_subtract)
    res = accumulate.to_dict()
    assert res == {
        "type": "haystack.testing.sample_components.accumulate.Accumulate",
        "init_parameters": {"function": "test_accumulate.my_subtract"},
    }


def test_from_dict():
    data = {"type": "haystack.testing.sample_components.accumulate.Accumulate", "init_parameters": {}}
    accumulate = Accumulate.from_dict(data)
    assert accumulate.function == _default_function


def test_from_dict_with_default_function():
    data = {
        "type": "haystack.testing.sample_components.accumulate.Accumulate",
        "init_parameters": {"function": "haystack.testing.sample_components.accumulate._default_function"},
    }
    accumulate = Accumulate.from_dict(data)
    assert accumulate.function == _default_function


def test_from_dict_with_custom_function():
    data = {
        "type": "haystack.testing.sample_components.accumulate.Accumulate",
        "init_parameters": {"function": "test_accumulate.my_subtract"},
    }
    accumulate = Accumulate.from_dict(data)
    assert accumulate.function == my_subtract


def test_accumulate_default():
    component = Accumulate()
    results = component.run(value=10)
    assert results == {"value": 10}
    assert component.state == 10

    results = component.run(value=1)
    assert results == {"value": 11}
    assert component.state == 11


def test_accumulate_callable():
    component = Accumulate(function=my_subtract)

    results = component.run(value=10)
    assert results == {"value": -10}
    assert component.state == -10

    results = component.run(value=1)
    assert results == {"value": -11}
    assert component.state == -11
