# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from sample_components import AddFixedValue
from canals.serialization import component_to_dict, component_from_dict


def test_to_dict():
    component = AddFixedValue()
    res = component_to_dict(component)
    assert res == {"type": "AddFixedValue", "init_parameters": {"add": 1}}


def test_to_dict_with_custom_add_value():
    component = AddFixedValue(add=100)
    res = component_to_dict(component)
    assert res == {"type": "AddFixedValue", "init_parameters": {"add": 100}}


def test_from_dict():
    data = {"type": "AddFixedValue"}
    component = component_from_dict(AddFixedValue, data)
    assert component.add == 1


def test_from_dict_with_custom_add_value():
    data = {"type": "AddFixedValue", "init_parameters": {"add": 100}}
    component = component_from_dict(AddFixedValue, data)
    assert component.add == 100


def test_run():
    component = AddFixedValue()
    results = component.run(value=50, add=10)
    assert results == {"result": 60}
