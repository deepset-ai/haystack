from typing import Any, Optional
import logging

from dataclasses import dataclass

from canals import component


logger = logging.getLogger(__name__)


@component
class Greet:
    """
    Logs a greeting message without affecting the value
    passing on the connection.

    Single input, single output component.
    """

    @dataclass
    class Output:
        value: int

    def __init__(self, message: str = "\nGreeting component says: Hi! The value is {value}\n", log_level: str = "INFO"):
        """
        :param message: the message to log. Can use {value} to embed the value.
        """
        self.message = message
        self.log_level = getattr(logging, log_level, None)
        if not self.log_level:
            raise ValueError(f"This log level does not exist: {log_level}")

    def run(self, value: int, message: Optional[str] = None, log_level: Optional[str] = None) -> Output:
        if log_level:
            new_log_level = getattr(logging, log_level, None)
            if not new_log_level:
                raise ValueError(f"This log level does not exist: {log_level}")
            level = new_log_level
        else:
            level = self.log_level

        message = message or self.message

        logger.log(level=level, msg=message.format(value=value))
        return Greet.Output(value=value)


def test_greet_default(caplog):
    caplog.set_level(logging.INFO)
    component = Greet()
    results = component.run(value=10)
    assert results == Greet.Output(value=10)
    assert "\nGreeting component says: Hi! The value is 10\n" in caplog.text
    assert component.init_parameters == {}


def test_greet_message_in_init(caplog):
    caplog.set_level(logging.INFO)
    component = Greet(message="Hello, that's {value}")
    results = component.run(value=10)
    assert results == Greet.Output(value=10)
    assert "Hello, that's 10" in caplog.text
    assert component.init_parameters == {"message": "Hello, that's {value}"}


def test_greet_message_in_run(caplog):
    caplog.set_level(logging.INFO)
    component = Greet()
    results = component.run(value=10, message="Hello, that's {value}")
    assert results == Greet.Output(value=10)
    assert "Hello, that's 10" in caplog.text
    assert component.init_parameters == {}


def test_greet_level_in_init(caplog):
    caplog.set_level(logging.WARNING)
    component = Greet(log_level="WARNING")
    results = component.run(value=10)
    assert results == Greet.Output(value=10)
    assert "\nGreeting component says: Hi! The value is 10\n" in caplog.text
    assert component.init_parameters == {"log_level": "WARNING"}


def test_greet_level_in_run(caplog):
    caplog.set_level(logging.WARNING)
    component = Greet()
    results = component.run(value=10, log_level="WARNING")
    assert results == Greet.Output(value=10)
    assert "\nGreeting component says: Hi! The value is 10\n" in caplog.text
    assert component.init_parameters == {}
