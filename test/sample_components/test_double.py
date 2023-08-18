# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from canals.testing import BaseTestComponent
from sample_components import Double


class TestDouble(BaseTestComponent):
    def test_to_dict(self):
        component = Double()
        res = component.to_dict()
        assert res == {"hash": id(component), "type": "Double", "init_parameters": {}}

    def test_from_dict(self):
        data = {"hash": 12345, "type": "Double", "init_parameters": {}}
        component = Double.from_dict(data)
        assert component

    def test_double_default(self):
        component = Double()
        results = component.run(value=10)
        assert results == {"value": 20}
