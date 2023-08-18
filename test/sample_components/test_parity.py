# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from canals.testing import BaseTestComponent
from sample_components import Parity


class TestParity(BaseTestComponent):
    def test_to_dict(self):
        component = Parity()
        res = component.to_dict()
        assert res == {"hash": id(component), "type": "Parity", "init_parameters": {}}

    def test_from_dict(self):
        data = {"hash": 12345, "type": "Parity", "init_parameters": {}}
        component = Parity.from_dict(data)
        assert component

    def test_parity(self):
        component = Parity()
        results = component.run(value=1)
        assert results == {"odd": 1, "even": None}
        results = component.run(value=2)
        assert results == {"odd": None, "even": 2}
