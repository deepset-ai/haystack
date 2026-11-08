# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack.core.serialization import default_to_dict
from haystack.dataclasses import ChatMessage
from haystack.token_counters.types import TokenCounter
from haystack.token_counters.utils import _non_text_tokens, _rendered_conversation, _rendered_tools
from haystack.tools import ToolsType


class ApproximateTokenCounter(TokenCounter):
    """
    Estimates tokens from text length using a flat ratio of characters to tokens.

    ## Usage Example:
    ```python
    from haystack.dataclasses import ChatMessage
    from haystack.token_counters import ApproximateTokenCounter

    counter = ApproximateTokenCounter(chars_per_token=4.0)
    messages = [
        ChatMessage.from_user("Hello, how are you?"),
        ChatMessage.from_assistant("I'm good, thank you! How can I assist you today?")
    ]
    token_count = counter.count(messages)
    print(f"Estimated token count: {token_count}")
    ```
    """

    def __init__(self, chars_per_token: float = 4.0, tokens_per_image: int = 85, tokens_per_file: int = 1000) -> None:
        """
        Initialize the counter.

        :param chars_per_token: How many characters to treat as one token.
        :param tokens_per_image: Tokens to charge per image, which has no text to measure. The default is what
            OpenAI charges for a small image; raise it if you send large ones.
        :param tokens_per_file: Tokens to charge per file. A rough stand-in for a short document, since the real
            cost depends on the page count; raise it if you send long ones.
        :raises ValueError: If `chars_per_token` is not positive.
        """
        if chars_per_token <= 0:
            raise ValueError(f"`chars_per_token` must be greater than 0, got {chars_per_token}.")
        self.chars_per_token = chars_per_token
        self.tokens_per_image = tokens_per_image
        self.tokens_per_file = tokens_per_file

    def count(self, messages: list[ChatMessage], tools: ToolsType | None = None) -> int:
        """
        Return the estimated number of tokens the given messages occupy.

        :param messages: The messages to measure.
        :param tools: Tools whose schemas are sent alongside the messages, and so consume tokens too.
        :returns: The estimated token count, or `0` when there is nothing to measure.
        """
        if not messages and not tools:
            return 0
        text = _rendered_conversation(messages) + _rendered_tools(tools)
        text_tokens = int(len(text) / self.chars_per_token)
        return text_tokens + _non_text_tokens(
            messages=messages, tokens_per_image=self.tokens_per_image, tokens_per_file=self.tokens_per_file
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the counter.

        :returns: A dictionary representation of the counter.
        """
        return default_to_dict(
            self,
            chars_per_token=self.chars_per_token,
            tokens_per_image=self.tokens_per_image,
            tokens_per_file=self.tokens_per_file,
        )
