# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Optional

import haystack.logging as haystack_logging
from haystack.core.component import component

logger = haystack_logging.getLogger(__name__)


@component
class Greet:
    """
    Logs a greeting message without affecting the value passing on the connection.
    """

    def __init__(self, message: str = "\nGreeting component says: Hi! The value is {value}\n", log_level: str = "INFO"):
        """
        Class constructor

        :param message: the message to log. Can use `{value}` to embed the value.
        :param log_level: the level to log at.
        """
        if log_level and not getattr(logging, log_level):
            raise ValueError(f"This log level does not exist: {log_level}")
        self.message = message
        self.log_level = log_level

    @component.output_types(value=int)
    def run(self, value: int, message: Optional[str] = None, log_level: Optional[str] = None):
        """
        Logs a greeting message without affecting the value passing on the connection.
        """
        if not message:
            message = self.message
        if not log_level:
            log_level = self.log_level

        level = getattr(logging, log_level, None)
        if not level:
            raise ValueError(f"This log level does not exist: {log_level}")

        logger.log(level=level, msg=message.format(value=value))
        return {"value": value}
