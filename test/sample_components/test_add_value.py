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

    def test_addvalue(self):
        component = AddFixedValue()
        results = component.run(value=50, add=10)
        assert results == {"result": 60}
        assert component.init_parameters == {}
