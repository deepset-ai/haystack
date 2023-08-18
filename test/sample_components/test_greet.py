# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import logging

from sample_components import Greet


def test_to_dict():
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


def test_to_dict_with_custom_parameters():
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


def test_from_dict():
    data = {
        "hash": 1234,
        "type": "Greet",
        "init_parameters": {},
    }
    component = Greet.from_dict(data)
    assert component.message == "\nGreeting component says: Hi! The value is {value}\n"
    assert component.log_level == "INFO"


def test_from_with_custom_parameters():
    data = {
        "hash": 1234,
        "type": "Greet",
        "init_parameters": {"message": "My message", "log_level": "ERROR"},
    }
    component = Greet.from_dict(data)
    assert component.message == "My message"
    assert component.log_level == "ERROR"


def test_greet_message(caplog):
    caplog.set_level(logging.WARNING)
    component = Greet()
    results = component.run(value=10, message="Hello, that's {value}", log_level="WARNING")
    assert results == {"value": 10}
    assert "Hello, that's 10" in caplog.text
