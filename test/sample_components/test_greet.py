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

    def test_to_dict(self):
        component = Greet()
        res = component.to_dict()
        assert res == {
            "hash": id(component),
            "type": "Greet",
            "init_parameters": {
                "message": "\nGreeting component says: Hi! The value is {value}\n",
                "log_level": "INFO",
            },
        }

    def test_to_dict_with_custom_parameters(self):
        component = Greet(message="My message", log_level="ERROR")
        res = component.to_dict()
        assert res == {
            "hash": id(component),
            "type": "Greet",
            "init_parameters": {
                "message": "My message",
                "log_level": "ERROR",
            },
        }

    def test_from_dict(self):
        data = {
            "hash": 1234,
            "type": "Greet",
            "init_parameters": {},
        }
        component = Greet.from_dict(data)
        assert component.message == "\nGreeting component says: Hi! The value is {value}\n"
        assert component.log_level == "INFO"

    def test_from_with_custom_parameters(self):
        data = {
            "hash": 1234,
            "type": "Greet",
            "init_parameters": {"message": "My message", "log_level": "ERROR"},
        }
        component = Greet.from_dict(data)
        assert component.message == "My message"
        assert component.log_level == "ERROR"

    def test_greet_message(self, caplog):
        caplog.set_level(logging.WARNING)
        component = Greet()
        results = component.run(value=10, message="Hello, that's {value}", log_level="WARNING")
        assert results == {"value": 10}
        assert "Hello, that's 10" in caplog.text
