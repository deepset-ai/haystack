# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import pytest

from canals.testing import BaseTestComponent
from sample_components import Remainder


class TestRemainder(BaseTestComponent):
    def test_saveload_default(self, tmp_path):
        self.assert_can_be_saved_and_loaded_in_pipeline(Remainder(), tmp_path)

    def test_saveload_divisor(self, tmp_path):
        self.assert_can_be_saved_and_loaded_in_pipeline(Remainder(divisor=1), tmp_path)

    def test_remainder_default(self):
        component = Remainder()
        results = component.run(component.input(value=3))
        assert results == component.output(remainder_is_1=3)

    def test_remainder_with_divisor(self):
        component = Remainder(divisor=4)
        results = component.run(component.input(value=3))
        assert results == component.output(remainder_is_3=3)

    def test_remainder_zero(self):
        with pytest.raises(ValueError):
            Remainder(divisor=0)
