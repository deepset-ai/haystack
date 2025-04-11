# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Protocol, TypeVar

from haystack.dataclasses import ChatMessage

# Ellipsis are needed to define the Protocol but pylint complains. See https://github.com/pylint-dev/pylint/issues/9319.
# pylint: disable=unnecessary-ellipsis

T = TypeVar("T", bound="ChatGenerator")


class ChatGenerator(Protocol):
    """
    Protocol for Chat Generators.

    This protocol defines the minimal interface that Chat Generators must implement.
    Chat Generators are components that process a list of `ChatMessage` objects as input and generate
    responses using a Language Model. They return a dictionary.
    """

    def run(self, messages: List[ChatMessage]) -> Dict[str, Any]:
        """
        Generate messages using the underlying Language Model.

        Implementing classes may accept additional optional parameters in their run method.
        For example: `def run (self, messages: List[ChatMessage], param_a="default", param_b="another_default")`.

        :param messages:
            A list of ChatMessage instances representing the input messages.
        :returns:
            A dictionary.
        """
        ...
