# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from sample_components import Sum
from canals.serialization import component_to_dict, component_from_dict


def test_to_dict():
    component = Sum()
    res = component_to_dict(component)
    assert res == {"type": "Sum", "init_parameters": {}}


def test_from_dict():
    data = {"type": "Sum", "init_parameters": {}}
    component = component_from_dict(Sum, data)


def test_sum_receives_no_values():
    component = Sum()
    results = component.run(values=[])
    assert results == {"total": 0}


def test_sum_receives_one_value():
    component = Sum()
    assert component.run(values=[10]) == {"total": 10}


def test_sum_receives_few_values():
    component = Sum()
    assert component.run(values=[10, 2]) == {"total": 12}
