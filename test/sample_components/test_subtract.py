# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from canals.testing import BaseTestComponent
from sample_components import Subtract


class TestSubtract(BaseTestComponent):
    def test_saveload_default(self, tmp_path):
        self.assert_can_be_saved_and_loaded_in_pipeline(Subtract(), tmp_path)

    def test_subtract(self):
        component = Subtract()
        results = component.run(first_value=10, second_value=7)
        assert results == {"difference": 3}
