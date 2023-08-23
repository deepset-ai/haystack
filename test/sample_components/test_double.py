# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from sample_components import Double


def test_to_dict():
    component = Double()
    res = component.to_dict()
    assert res == {"type": "Double", "init_parameters": {}}


def test_from_dict():
    data = {"type": "Double", "init_parameters": {}}
    component = Double.from_dict(data)
    assert component


def test_double_default():
    component = Double()
    results = component.run(value=10)
    assert results == {"value": 20}
