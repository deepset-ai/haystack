# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from sample_components import Threshold
from canals.serialization import component_to_dict, component_from_dict


def test_to_dict():
    component = Threshold()
    res = component_to_dict(component)
    assert res == {"type": "Threshold", "init_parameters": {"threshold": 10}}


def test_to_dict_with_custom_threshold_value():
    component = Threshold(threshold=100)
    res = component_to_dict(component)
    assert res == {"type": "Threshold", "init_parameters": {"threshold": 100}}


def test_from_dict():
    data = {"type": "Threshold"}
    component = component_from_dict(Threshold, data)
    assert component.threshold == 10


def test_from_dict_with_custom_threshold_value():
    data = {"type": "Threshold", "init_parameters": {"threshold": 100}}
    component = component_from_dict(Threshold, data)
    assert component.threshold == 100


def test_threshold():
    component = Threshold()

    results = component.run(value=5, threshold=10)
    assert results == {"below": 5}

    results = component.run(value=15, threshold=10)
    assert results == {"above": 15}
