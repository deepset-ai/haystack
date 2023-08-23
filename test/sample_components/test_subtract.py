# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from sample_components import Subtract


def test_to_dict():
    component = Subtract()
    res = component.to_dict()
    assert res == {"type": "Subtract", "init_parameters": {}}


def test_from_dict():
    data = {"type": "Subtract", "init_parameters": {}}
    component = Subtract.from_dict(data)
    assert component


def test_subtract():
    component = Subtract()
    results = component.run(first_value=10, second_value=7)
    assert results == {"difference": 3}
