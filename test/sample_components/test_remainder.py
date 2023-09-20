# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import pytest

from sample_components import Remainder
from canals.serialization import component_to_dict, component_from_dict


def test_to_dict():
    component = Remainder()
    res = component_to_dict(component)
    assert res == {"type": "Remainder", "init_parameters": {"divisor": 3}}


def test_to_dict_with_custom_divisor_value():
    component = Remainder(divisor=100)
    res = component_to_dict(component)
    assert res == {"type": "Remainder", "init_parameters": {"divisor": 100}}


def test_from_dict():
    data = {"type": "Remainder"}
    component = component_from_dict(Remainder, data)
    assert component.divisor == 3


def test_from_dict_with_custom_divisor_value():
    data = {"type": "Remainder", "init_parameters": {"divisor": 100}}
    component = component_from_dict(Remainder, data)
    assert component.divisor == 100


def test_remainder_default():
    component = Remainder()
    results = component.run(value=4)
    assert results == {"remainder_is_0": None, "remainder_is_1": 4, "remainder_is_2": None}


def test_remainder_with_divisor():
    component = Remainder(divisor=4)
    results = component.run(value=4)
    assert results == {"remainder_is_0": 4, "remainder_is_1": None, "remainder_is_2": None, "remainder_is_3": None}


def test_remainder_zero():
    with pytest.raises(ValueError):
        Remainder(divisor=0)
