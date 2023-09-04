# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import logging

from sample_components import Greet
from canals.serialization import component_to_dict, component_from_dict


def test_to_dict():
    component = Greet()
    res = component_to_dict(component)
    assert res == {
        "type": "Greet",
        "init_parameters": {
            "message": "\nGreeting component says: Hi! The value is {value}\n",
            "log_level": "INFO",
        },
    }


def test_to_dict_with_custom_parameters():
    component = Greet(message="My message", log_level="ERROR")
    res = component_to_dict(component)
    assert res == {
        "type": "Greet",
        "init_parameters": {
            "message": "My message",
            "log_level": "ERROR",
        },
    }


def test_from_dict():
    data = {
        "type": "Greet",
        "init_parameters": {},
    }
    component = component_from_dict(Greet, data)
    assert component.message == "\nGreeting component says: Hi! The value is {value}\n"
    assert component.log_level == "INFO"


def test_from_with_custom_parameters():
    data = {
        "type": "Greet",
        "init_parameters": {"message": "My message", "log_level": "ERROR"},
    }
    component = component_from_dict(Greet, data)
    assert component.message == "My message"
    assert component.log_level == "ERROR"


def test_greet_message(caplog):
    caplog.set_level(logging.WARNING)
    component = Greet()
    results = component.run(value=10, message="Hello, that's {value}", log_level="WARNING")
    assert results == {"value": 10}
    assert "Hello, that's 10" in caplog.text
