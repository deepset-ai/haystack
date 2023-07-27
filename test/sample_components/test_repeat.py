# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from canals.testing import BaseTestComponent
from sample_components import Repeat


class TestRepeat(BaseTestComponent):
    def test_saveload(self, tmp_path):
        self.assert_can_be_saved_and_loaded_in_pipeline(Repeat(outputs=["one", "two"]), tmp_path)

    def test_repeat_default(self):
        component = Repeat(outputs=["one", "two"])
        results = component.run(value=10)
        assert results == {"one": 10, "two": 10}
        assert component.init_parameters == {"outputs": ["one", "two"]}
