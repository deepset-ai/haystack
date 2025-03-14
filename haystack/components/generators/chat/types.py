# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Protocol, TypeVar

from haystack.dataclasses import ChatMessage

T = TypeVar("T", bound="ChatGenerator")


class ChatGenerator(Protocol):
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :returns:
            The serialized component as a dictionary.
        """
        ...

    @classmethod
    def from_dict(cls: type[T], data: Dict[str, Any]) -> T:
        """
        Deserialize this component from a dictionary.

        Returns an instance of the specific implementing class.
        """
        ...

    def run(self, messages: List[ChatMessage]) -> Dict[str, Any]:
        """
        Run the chat generator.
        """
        ...
