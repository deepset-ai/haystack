# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from sample_components import Parity


def test_to_dict():
    component = Parity()
    res = component.to_dict()
    assert res == {"hash": id(component), "type": "Parity", "init_parameters": {}}


def test_from_dict():
    data = {"hash": 12345, "type": "Parity", "init_parameters": {}}
    component = Parity.from_dict(data)
    assert component


def test_parity():
    component = Parity()
    results = component.run(value=1)
    assert results == {"odd": 1, "even": None}
    results = component.run(value=2)
    assert results == {"odd": None, "even": 2}
