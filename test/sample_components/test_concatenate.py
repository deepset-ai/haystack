# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from sample_components import Concatenate
from canals.serialization import component_to_dict, component_from_dict


def test_to_dict():
    component = Concatenate()
    res = component_to_dict(component)
    assert res == {"type": "Concatenate", "init_parameters": {}}


def test_from_dict():
    data = {"type": "Concatenate", "init_parameters": {}}
    component = component_from_dict(Concatenate, data)
    assert component


def test_input_lists():
    component = Concatenate()
    res = component.run(first=["This"], second=["That"])
    assert res == {"value": ["This", "That"]}


def test_input_strings():
    component = Concatenate()
    res = component.run(first="This", second="That")
    assert res == {"value": ["This", "That"]}


def test_input_first_list_second_string():
    component = Concatenate()
    res = component.run(first=["This"], second="That")
    assert res == {"value": ["This", "That"]}


def test_input_first_string_second_list():
    component = Concatenate()
    res = component.run(first="This", second=["That"])
    assert res == {"value": ["This", "That"]}
