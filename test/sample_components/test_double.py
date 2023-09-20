# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from sample_components import Double
from canals.serialization import component_to_dict, component_from_dict


def test_to_dict():
    component = Double()
    res = component_to_dict(component)
    assert res == {"type": "Double", "init_parameters": {}}


def test_from_dict():
    data = {"type": "Double", "init_parameters": {}}
    component = component_from_dict(Double, data)
    assert component


def test_double_default():
    component = Double()
    results = component.run(value=10)
    assert results == {"value": 20}
