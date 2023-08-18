# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from canals.testing import BaseTestComponent
from sample_components import Subtract


class TestSubtract(BaseTestComponent):
    def test_to_dict(self):
        component = Subtract()
        res = component.to_dict()
        assert res == {"hash": id(component), "type": "Subtract", "init_parameters": {}}

    def test_from_dict(self):
        data = {"hash": 12345, "type": "Subtract", "init_parameters": {}}
        component = Subtract.from_dict(data)
        assert component

    def test_subtract(self):
        component = Subtract()
        results = component.run(first_value=10, second_value=7)
        assert results == {"difference": 3}
