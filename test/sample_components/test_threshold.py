# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from canals.testing import BaseTestComponent
from sample_components import Threshold


class TestThreshold(BaseTestComponent):
    def test_saveload_default(self, tmp_path):
        self.assert_can_be_saved_and_loaded_in_pipeline(Threshold(), tmp_path)

    def test_saveload_threshold(self, tmp_path):
        self.assert_can_be_saved_and_loaded_in_pipeline(Threshold(threshold=3), tmp_path)

    def test_threshold(self):
        component = Threshold()

        results = component.run(value=5, threshold=10)
        assert results == {"above": None, "below": 5}

        results = component.run(value=15, threshold=10)
        assert results == {"above": 15, "below": None}
