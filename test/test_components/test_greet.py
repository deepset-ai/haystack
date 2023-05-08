import logging

from dataclasses import dataclass

import pytest

from canals.testing import BaseTestComponent
from canals import component


logger = logging.getLogger(__name__)


@component
class Greet:
    """
    Logs a greeting message without affecting the value passing on the connection.
    """

    @dataclass
    class Output:
        value: int

    def __init__(self, message: str = "\nGreeting component says: Hi! The value is {value}\n", log_level: str = "INFO"):
        """
        :param message: the message to log. Can use {value} to embed the value.
        :param log_level: the level to log at.
        """
        if not getattr(logging, log_level):
            raise ValueError(f"This log level does not exist: {log_level}")
        self.defaults = {"message": message, "log_level": log_level}

    def run(self, value: int, message: str, log_level: str) -> Output:
        """
        :param message: the message to log. Can use {value} to embed the value.
        :param log_level: the level to log at.
        """
        level = getattr(logging, log_level, None)
        if not level:
            raise ValueError(f"This log level does not exist: {log_level}")

        logger.log(level=level, msg=message.format(value=value))
        return Greet.Output(value=value)


class TestGreet(BaseTestComponent):
    @pytest.fixture
    def components(self):
        return [
            Greet(),
            Greet(message="Hello, that's {value}"),
            Greet(log_level="WARNING"),
            Greet(message="Hello, that's {value}", log_level="WARNING"),
        ]

    def test_greet_message(self, caplog):
        caplog.set_level(logging.WARNING)
        component = Greet()
        results = component.run(value=10, message="Hello, that's {value}", log_level="WARNING")
        assert results == Greet.Output(value=10)
        assert "Hello, that's 10" in caplog.text
