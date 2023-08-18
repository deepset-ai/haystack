# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from sample_components import Repeat


def test_to_dict():
    component = Repeat(outputs=["first", "second"])
    res = component.to_dict()
    assert res == {"hash": id(component), "type": "Repeat", "init_parameters": {"outputs": ["first", "second"]}}


def test_from_dict():
    data = {"hash": 1234, "type": "Repeat", "init_parameters": {"outputs": ["first", "second"]}}
    component = Repeat.from_dict(data)
    assert component.outputs == ["first", "second"]


def test_repeat_default():
    component = Repeat(outputs=["one", "two"])
    results = component.run(value=10)
    assert results == {"one": 10, "two": 10}
