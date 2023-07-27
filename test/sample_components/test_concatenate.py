# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from canals.testing import BaseTestComponent
from sample_components import Concatenate


class TestConcatenate(BaseTestComponent):
    def test_saveload_default(self, tmp_path):
        self.assert_can_be_saved_and_loaded_in_pipeline(Concatenate(), tmp_path)

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
