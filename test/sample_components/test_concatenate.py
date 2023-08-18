# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from canals.testing import BaseTestComponent
from sample_components import Concatenate


class TestConcatenate(BaseTestComponent):
    def test_to_dict(self):
        component = Concatenate()
        res = component.to_dict()
        assert res == {"hash": id(component), "type": "Concatenate", "init_parameters": {}}

    def test_from_dict(self):
        data = {"hash": 12345, "type": "Concatenate", "init_parameters": {}}
        component = Concatenate.from_dict(data)
        assert component

    def test_input_lists(self):
        component = Concatenate()
        res = component.run(first=["This"], second=["That"])
        assert res == {"value": ["This", "That"]}

    def test_input_strings(self):
        component = Concatenate()
        res = component.run(first="This", second="That")
        assert res == {"value": ["This", "That"]}

    def test_input_first_list_second_string(self):
        component = Concatenate()
        res = component.run(first=["This"], second="That")
        assert res == {"value": ["This", "That"]}

    def test_input_first_string_second_list(self):
        component = Concatenate()
        res = component.run(first="This", second=["That"])
        assert res == {"value": ["This", "That"]}
