# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from sample_components import AddFixedValue


def test_to_dict():
    component = AddFixedValue()
    res = component.to_dict()
    assert res == {"hash": id(component), "type": "AddFixedValue", "init_parameters": {"add": 1}}


def test_to_dict_with_custom_add_value():
    component = AddFixedValue(add=100)
    res = component.to_dict()
    assert res == {"hash": id(component), "type": "AddFixedValue", "init_parameters": {"add": 100}}


def test_from_dict():
    data = {"hash": 1234, "type": "AddFixedValue"}
    component = AddFixedValue.from_dict(data)
    assert component.add == 1


def test_from_dict_with_custom_add_value():
    data = {"hash": 1234, "type": "AddFixedValue", "init_parameters": {"add": 100}}
    component = AddFixedValue.from_dict(data)
    assert component.add == 100


def test_run():
    component = AddFixedValue()
    results = component.run(value=50, add=10)
    assert results == {"result": 60}
