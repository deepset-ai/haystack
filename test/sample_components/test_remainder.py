# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import pytest

from canals.testing import BaseTestComponent
from sample_components import Remainder


class TestRemainder(BaseTestComponent):
    def test_to_dict(self):
        component = Remainder()
        res = component.to_dict()
        assert res == {"hash": id(component), "type": "Remainder", "init_parameters": {"divisor": 3}}

    def test_to_dict_with_custom_divisor_value(self):
        component = Remainder(divisor=100)
        res = component.to_dict()
        assert res == {"hash": id(component), "type": "Remainder", "init_parameters": {"divisor": 100}}

    def test_from_dict(self):
        data = {"hash": 1234, "type": "Remainder"}
        component = Remainder.from_dict(data)
        assert component.divisor == 3

    def test_from_dict_with_custom_divisor_value(self):
        data = {"hash": 1234, "type": "Remainder", "init_parameters": {"divisor": 100}}
        component = Remainder.from_dict(data)
        assert component.divisor == 100

    def test_remainder_default(self):
        component = Remainder()
        results = component.run(value=4)
        assert results == {"remainder_is_0": None, "remainder_is_1": 4, "remainder_is_2": None}

    def test_remainder_with_divisor(self):
        component = Remainder(divisor=4)
        results = component.run(value=4)
        assert results == {"remainder_is_0": 4, "remainder_is_1": None, "remainder_is_2": None, "remainder_is_3": None}

    def test_remainder_zero(self):
        with pytest.raises(ValueError):
            Remainder(divisor=0)
