# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from canals.testing import BaseTestComponent
from sample_components import AddFixedValue


class TestAddFixedValue(BaseTestComponent):
    def test_saveload_default(self, tmp_path):
        self.assert_can_be_saved_and_loaded_in_pipeline(AddFixedValue(), tmp_path)

    def test_saveload_add(self, tmp_path):
        self.assert_can_be_saved_and_loaded_in_pipeline(AddFixedValue(add=2), tmp_path)

    def test_to_dict(self):
        component = AddFixedValue()
        res = component.to_dict()
        assert res == {"hash": id(component), "type": "AddFixedValue", "init_parameters": {"add": 1}}

    def test_to_dict_with_custom_add_value(self):
        component = AddFixedValue(add=100)
        res = component.to_dict()
        assert res == {"hash": id(component), "type": "AddFixedValue", "init_parameters": {"add": 100}}

    def test_from_dict(self):
        data = {"hash": 1234, "type": "AddFixedValue"}
        component = AddFixedValue.from_dict(data)
        assert component.add == 1

    def test_from_dict_with_custom_add_value(self):
        data = {"hash": 1234, "type": "AddFixedValue", "init_parameters": {"add": 100}}
        component = AddFixedValue.from_dict(data)
        assert component.add == 100

    def test_run(self):
        component = AddFixedValue()
        results = component.run(value=50, add=10)
        assert results == {"result": 60}
