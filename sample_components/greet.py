# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Optional
import logging


from canals import component


logger = logging.getLogger(__name__)


@component
class Greet:
    """
    Logs a greeting message without affecting the value passing on the connection.
    """

    @component.input
    def input(self):
        class Input:
            value: int
            message: str
            log_level: str

        return Input

    @component.output
    def output(self):
        class Output:
            value: int

        return Output

    def __init__(
        self,
        message: Optional[str] = "\nGreeting component says: Hi! The value is {value}\n",
        log_level: Optional[str] = "INFO",
    ):
        """
        :param message: the message to log. Can use `{value}` to embed the value.
        :param log_level: the level to log at.
        """
        if log_level and not getattr(logging, log_level):
            raise ValueError(f"This log level does not exist: {log_level}")
        self.defaults = {"message": message, "log_level": log_level}

    def run(self, data):
        """
        Logs a greeting message without affecting the value passing on the connection.
        """
        print(data.log_level)
        level = getattr(logging, data.log_level, None)
        if not level:
            raise ValueError(f"This log level does not exist: {data.log_level}")

        logger.log(level=level, msg=data.message.format(value=data.value))
        return self.output(value=data.value)
