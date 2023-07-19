# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from canals.testing import BaseTestComponent
from sample_components import Parity


class TestParity(BaseTestComponent):
    def test_saveload_default(self, tmp_path):
        self.assert_can_be_saved_and_loaded_in_pipeline(Parity(), tmp_path)

    def test_parity(self):
        component = Parity()
        results = component.run(component.input(value=1))
        assert results == component.output(odd=1)
        results = component.run(component.input(value=2))
        assert results == component.output(even=2)
