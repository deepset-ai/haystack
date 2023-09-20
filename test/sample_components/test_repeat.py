# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from sample_components import Repeat
from canals.serialization import component_to_dict, component_from_dict


def test_to_dict():
    component = Repeat(outputs=["first", "second"])
    res = component_to_dict(component)
    assert res == {"type": "Repeat", "init_parameters": {"outputs": ["first", "second"]}}


def test_from_dict():
    data = {"type": "Repeat", "init_parameters": {"outputs": ["first", "second"]}}
    component = component_from_dict(Repeat, data)
    assert component.outputs == ["first", "second"]


def test_repeat_default():
    component = Repeat(outputs=["one", "two"])
    results = component.run(value=10)
    assert results == {"one": 10, "two": 10}
