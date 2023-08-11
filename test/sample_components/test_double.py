# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from canals.testing import BaseTestComponent
from sample_components import Double


class TestDouble(BaseTestComponent):
    def test_saveload_default(self, tmp_path):
        self.assert_can_be_saved_and_loaded_in_pipeline(Double(), tmp_path)

    def test_double_default(self):
        component = Double()
        results = component.run(value=10)
        assert results == {"value": 20}
