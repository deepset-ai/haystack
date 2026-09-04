# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any

from openai import OpenAI

from haystack.components.generators.chat.openai_responses import _convert_chat_message_to_responses_api_format
from haystack.core.serialization import default_to_dict
from haystack.dataclasses import ChatMessage
from haystack.token_counters.types import TokenCounter
from haystack.tools import ToolsType, flatten_tools_or_toolsets
from haystack.utils import Secret
from haystack.utils.http_client import init_http_client


class OpenAITokenCounter(TokenCounter):
    """
    Counts tokens with OpenAI's input token counting API.

    Unlike local token counters, this counter sends the input to OpenAI's
    `POST /v1/responses/input_tokens` endpoint. The returned count includes the model-specific formatting used for
    messages and tool schemas, as well as supported non-text content such as images and files.

    ## Usage Example:
    ```python
    from haystack.dataclasses import ChatMessage
    from haystack.token_counters import OpenAITokenCounter

    counter = OpenAITokenCounter("gpt-5-mini")
    messages = [ChatMessage.from_user("Hello, how are you?")]
    token_count = counter.count(messages)
    print(f"Token count: {token_count}")
    ```
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: Secret = Secret.from_env_var("OPENAI_API_KEY"),
        api_base_url: str | None = None,
        organization: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        http_client_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the counter.

        :param model: The model whose tokenization should be used.
        :param api_key: The OpenAI API key. You can set it with the `OPENAI_API_KEY` environment variable or pass it
            explicitly.
        :param api_base_url: An optional base URL for the OpenAI API.
        :param organization: Your OpenAI organization ID.
        :param timeout: Timeout for OpenAI client calls. If unset, uses `OPENAI_TIMEOUT` or 30 seconds.
        :param max_retries: Maximum retries for OpenAI client calls. If unset, uses `OPENAI_MAX_RETRIES` or 5.
        :param http_client_kwargs: Keyword arguments used to configure the underlying HTTPX client.
        """
        self.api_key = api_key
        self.model = model
        self.api_base_url = api_base_url
        self.organization = organization
        self.timeout = timeout
        self.max_retries = max_retries
        self.http_client_kwargs = http_client_kwargs

        self.client: OpenAI | None = None

    def warm_up(self) -> None:
        """Initialize the OpenAI client."""
        if self.client is not None:
            return

        timeout = self.timeout if self.timeout is not None else float(os.environ.get("OPENAI_TIMEOUT", "30.0"))
        max_retries = (
            self.max_retries if self.max_retries is not None else int(os.environ.get("OPENAI_MAX_RETRIES", "5"))
        )
        # openai>=3 annotates http_client as httpx2, but legacy httpx clients are supported at runtime.
        # https://github.com/openai/openai-python/blob/main/httpx2.md
        http_client = init_http_client(self.http_client_kwargs, async_client=False)
        self.client = OpenAI(
            api_key=self.api_key.resolve_value(),
            organization=self.organization,
            base_url=self.api_base_url,
            timeout=timeout,
            max_retries=max_retries,
            http_client=http_client,  # type: ignore[arg-type]
        )

    def count(self, messages: list[ChatMessage], tools: ToolsType | None = None) -> int:
        """
        Return the exact number of input tokens OpenAI will use for the given messages and tools.

        :param messages: The messages to measure.
        :param tools: Tools whose schemas are sent alongside the messages, and so consume tokens too.
        :returns: The token count, or `0` when there is nothing to measure.
        """
        if not messages and not tools:
            return 0

        self.warm_up()
        client = self.client
        if client is None:
            raise RuntimeError("The OpenAI client was not initialized.")

        openai_input: list[dict[str, Any]] = []
        for message in messages:
            openai_input.extend(_convert_chat_message_to_responses_api_format(message=message))

        request: dict[str, Any] = {"model": self.model, "input": openai_input}
        if tools:
            request["tools"] = [
                {"type": "function", **tool.tool_spec} for tool in flatten_tools_or_toolsets(tools=tools)
            ]

        response = client.responses.input_tokens.count(**request)
        return response.input_tokens

    def close(self) -> None:
        """Close the OpenAI client and its underlying HTTP resources."""
        if self.client is not None:
            self.client.close()
            self.client = None

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the counter.

        :returns: A dictionary representation of the counter.
        """
        return default_to_dict(
            self,
            api_key=self.api_key,
            model=self.model,
            api_base_url=self.api_base_url,
            organization=self.organization,
            timeout=self.timeout,
            max_retries=self.max_retries,
            http_client_kwargs=self.http_client_kwargs,
        )
