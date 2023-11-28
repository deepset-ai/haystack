# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import pytest

from haystack.testing.sample_components import Remainder
from haystack.core.serialization import component_to_dict, component_from_dict


def test_remainder_default():
    component = Remainder()
    results = component.run(value=4)
    assert results == {"remainder_is_1": 4}


def test_remainder_with_divisor():
    component = Remainder(divisor=4)
    results = component.run(value=4)
    assert results == {"remainder_is_0": 4}


def test_remainder_zero():
    with pytest.raises(ValueError):
        Remainder(divisor=0)
