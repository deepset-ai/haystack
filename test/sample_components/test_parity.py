# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from sample_components import Parity
from canals.serialization import component_to_dict, component_from_dict


def test_to_dict():
    component = Parity()
    res = component_to_dict(component)
    assert res == {"type": "Parity", "init_parameters": {}}


def test_from_dict():
    data = {"type": "Parity", "init_parameters": {}}
    component = component_from_dict(Parity, data)
    assert component


def test_parity():
    component = Parity()
    results = component.run(value=1)
    assert results == {"odd": 1, "even": None}
    results = component.run(value=2)
    assert results == {"odd": None, "even": 2}
