# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from canals.testing import BaseTestComponent
from sample_components import Repeat


class TestRepeat(BaseTestComponent):
    def test_to_dict(self):
        component = Repeat(outputs=["first", "second"])
        res = component.to_dict()
        assert res == {"hash": id(component), "type": "Repeat", "init_parameters": {"outputs": ["first", "second"]}}

    def test_from_dict(self):
        data = {"hash": 1234, "type": "Repeat", "init_parameters": {"outputs": ["first", "second"]}}
        component = Repeat.from_dict(data)
        assert component.outputs == ["first", "second"]

    def test_repeat_default(self):
        component = Repeat(outputs=["one", "two"])
        results = component.run(value=10)
        assert results == {"one": 10, "two": 10}
