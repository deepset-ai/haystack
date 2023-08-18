# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from sample_components import Threshold


def test_to_dict():
    component = Threshold()
    res = component.to_dict()
    assert res == {"hash": id(component), "type": "Threshold", "init_parameters": {"threshold": 10}}


def test_to_dict_with_custom_threshold_value():
    component = Threshold(threshold=100)
    res = component.to_dict()
    assert res == {"hash": id(component), "type": "Threshold", "init_parameters": {"threshold": 100}}


def test_from_dict():
    data = {"hash": 1234, "type": "Threshold"}
    component = Threshold.from_dict(data)
    assert component.threshold == 10


def test_from_dict_with_custom_threshold_value():
    data = {"hash": 1234, "type": "Threshold", "init_parameters": {"threshold": 100}}
    component = Threshold.from_dict(data)
    assert component.threshold == 100


def test_threshold():
    component = Threshold()

    results = component.run(value=5, threshold=10)
    assert results == {"above": None, "below": 5}

    results = component.run(value=15, threshold=10)
    assert results == {"above": 15, "below": None}
