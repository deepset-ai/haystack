# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import logging

from canals.testing import BaseTestComponent
from sample_components import Greet


class TestGreet(BaseTestComponent):
    def test_saveload_default(self, tmp_path):
        self.assert_can_be_saved_and_loaded_in_pipeline(Greet(), tmp_path)

    def test_saveload_message(self, tmp_path):
        self.assert_can_be_saved_and_loaded_in_pipeline(Greet(message="Hello, that's {value}"), tmp_path)

    def test_saveload_loglevel(self, tmp_path):
        self.assert_can_be_saved_and_loaded_in_pipeline(Greet(log_level="WARNING"), tmp_path)

    def test_saveload_message_and_loglevel(self, tmp_path):
        self.assert_can_be_saved_and_loaded_in_pipeline(
            Greet(message="Hello, that's {value}", log_level="WARNING"), tmp_path
        )

    def test_greet_message(self, caplog):
        caplog.set_level(logging.WARNING)
        component = Greet()
        results = component.run(value=10, message="Hello, that's {value}", log_level="WARNING")
        assert results == {"value": 10}
        assert "Hello, that's 10" in caplog.text
