# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack.core.serialization import default_to_dict
from haystack.dataclasses import ChatMessage
from haystack.lazy_imports import LazyImport
from haystack.token_counters.types import TokenCounter
from haystack.token_counters.utils import _non_text_tokens, _rendered_conversation, _rendered_tools
from haystack.tools import ToolsType

with LazyImport("Run 'pip install tiktoken'") as tiktoken_imports:
    import tiktoken


class TiktokenCounter(TokenCounter):
    """
    Counts tokens locally with `tiktoken`, OpenAI's byte-pair encoder.

    Counting is an estimate, and two limits are worth knowing before relying on it:
    - **It is text-only**, so images and files get the flat `tokens_per_image` / `tokens_per_file` estimate rather
      than a real count.
    - **It is OpenAI's encoder.** Other providers tokenize differently, so expect the count to drift on them.

    ## Usage Example:
    ```python
    from haystack.dataclasses import ChatMessage
    from haystack.token_counters import TiktokenCounter

    counter = TiktokenCounter(encoding="o200k_base")
    messages = [
        ChatMessage.from_user("Hello, how are you?"),
        ChatMessage.from_assistant("I'm good, thank you! How can I assist you today?")
    ]
    token_count = counter.count(messages)
    print(f"Token count: {token_count}")
    ```
    """

    def __init__(self, encoding: str = "o200k_base", tokens_per_image: int = 85, tokens_per_file: int = 1000) -> None:
        """
        Initialize the counter.

        :param encoding: The `tiktoken` encoding to count with. The default, `o200k_base`, is what current OpenAI
            models use.
        :param tokens_per_image: Tokens to charge per image, which the tokenizer cannot measure. The default is what
            OpenAI charges for a small image; raise it if you send large ones.
        :param tokens_per_file: Tokens to charge per file. A rough stand-in for a short document, since the real
            cost depends on the page count; raise it if you send long ones.
        :raises ImportError: If `tiktoken` is not installed.
        """
        tiktoken_imports.check()
        self.encoding = encoding
        self.tokens_per_image = tokens_per_image
        self.tokens_per_file = tokens_per_file
        self._encoder: Any = None

    def warm_up(self) -> None:
        """Load the encoder, downloading its vocabulary if it is not already cached."""
        if self._encoder is not None:
            return
        self._encoder = tiktoken.get_encoding(self.encoding)

    def count(self, messages: list[ChatMessage], tools: ToolsType | None = None) -> int:
        """
        Return the estimated number of tokens used by the given messages.

        :param messages: The messages to measure.
        :param tools: Tools whose schemas are sent alongside the messages, and so consume tokens too.
        :returns: The estimated token count, or `0` when there is nothing to measure.
        """
        if not messages and not tools:
            return 0
        self.warm_up()
        text_tokens = len(self._encoder.encode(_rendered_conversation(messages) + _rendered_tools(tools)))
        return text_tokens + _non_text_tokens(
            messages, tokens_per_image=self.tokens_per_image, tokens_per_file=self.tokens_per_file
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the counter.

        :returns: A dictionary representation of the counter.
        """
        return default_to_dict(
            self, encoding=self.encoding, tokens_per_image=self.tokens_per_image, tokens_per_file=self.tokens_per_file
        )
