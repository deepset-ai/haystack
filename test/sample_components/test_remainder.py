# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import pytest

from sample_components import Remainder


def test_to_dict():
    component = Remainder()
    res = component.to_dict()
    assert res == {"hash": id(component), "type": "Remainder", "init_parameters": {"divisor": 3}}


def test_to_dict_with_custom_divisor_value():
    component = Remainder(divisor=100)
    res = component.to_dict()
    assert res == {"hash": id(component), "type": "Remainder", "init_parameters": {"divisor": 100}}


def test_from_dict():
    data = {"hash": 1234, "type": "Remainder"}
    component = Remainder.from_dict(data)
    assert component.divisor == 3


def test_from_dict_with_custom_divisor_value():
    data = {"hash": 1234, "type": "Remainder", "init_parameters": {"divisor": 100}}
    component = Remainder.from_dict(data)
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
