# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import io
import json
import os
import time
from typing import Any, Final

from openai import AsyncOpenAI, OpenAI
from openai.types import Batch

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.utils import Secret
from haystack.utils.http_client import init_http_client

logger = logging.getLogger(__name__)

# The only batch-supported endpoint for chat completions
_BATCH_ENDPOINT: Final = "/v1/chat/completions"

# Once a batch reaches one of these, there's nothing more to wait for
_TERMINAL_STATUSES = frozenset({"completed", "failed", "expired", "cancelled"})


@component
class OpenAIBatchChatGenerator:
    """
    Submits multiple conversations to OpenAI's Batch API for asynchronous processing.

    Unlike `OpenAIChatGenerator` which handles one request at a time, this component
    accepts *multiple* conversations, bundles them into a single batch job, and polls
    until OpenAI finishes processing. The trade-off is latency for cost: batch requests
    are 50% cheaper and enjoy higher rate limits, but can take up to 24 hours.

    Best suited for large-scale, non-latency-critical workloads like classification,
    summarization, or translation over thousands of inputs.

    ### Usage example
    ```python
    from haystack.components.generators.chat import OpenAIBatchChatGenerator
    from haystack.dataclasses import ChatMessage

    conversations = [
        [ChatMessage.from_user("Summarize: The quick brown fox...")],
        [ChatMessage.from_user("Summarize: To be or not to be...")],
        [ChatMessage.from_user("Summarize: It was the best of times...")],
    ]

    generator = OpenAIBatchChatGenerator()
    result = generator.run(message_sets=conversations)

    for reply_set in result["replies"]:
        print(reply_set[0].text)
    ```
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("OPENAI_API_KEY"),
        model: str = "gpt-5-mini",
        api_base_url: str | None = None,
        organization: str | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        poll_interval: float = 30.0,
        max_wait_seconds: float = 86400.0,
        completion_window: str = "24h",
        http_client_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Creates an instance of OpenAIBatchChatGenerator.

        Before initializing the component, you can set the 'OPENAI_TIMEOUT' and 'OPENAI_MAX_RETRIES'
        environment variables to override the `timeout` and `max_retries` parameters respectively
        in the OpenAI client.

        :param api_key: The OpenAI API key.
            You can set it with an environment variable `OPENAI_API_KEY`, or pass with this parameter
            during initialization.
        :param model: The name of the model to use for all requests in the batch.
        :param api_base_url: An optional base URL for the OpenAI API.
        :param organization: Your OpenAI organization ID, defaults to `None`. See
        [production best practices](https://platform.openai.com/docs/guides/production-best-practices/setting-up-your-organization).
        :param generation_kwargs: Default generation parameters (temperature, max_completion_tokens, etc.)
            applied to every request in the batch. These parameters are sent directly to the OpenAI endpoint.
            Can be overridden at runtime via the `generation_kwargs` parameter in `run()`.
        :param timeout: Timeout for the OpenAI SDK HTTP client (not the batch job itself).
            Defaults to the `OPENAI_TIMEOUT` env var, or 30 seconds.
        :param max_retries: Max retries for transient SDK errors.
            Defaults to the `OPENAI_MAX_RETRIES` env var, or 5.
        :param poll_interval: Seconds between batch status checks. Default: 30.
        :param max_wait_seconds: Maximum seconds to wait for batch completion before raising
            a TimeoutError. Default: 86400 (24 hours), matching OpenAI's completion window.
        :param completion_window: OpenAI's batch completion window. Currently only ``"24h"`` is supported
            by the API.
        :param http_client_kwargs: Keyword arguments for a custom ``httpx.Client`` or ``httpx.AsyncClient``.
            For more information, see the `HTTPX documentation <https://www.python-httpx.org/api/#client>`_.
        """
        self.api_key = api_key
        self.model = model
        self.api_base_url = api_base_url
        self.organization = organization
        self.generation_kwargs = generation_kwargs or {}
        self.timeout = timeout
        self.max_retries = max_retries
        self.poll_interval = poll_interval
        self.max_wait_seconds = max_wait_seconds
        self.completion_window = completion_window
        self.http_client_kwargs = http_client_kwargs

        self.client: OpenAI | None = None
        self.async_client: AsyncOpenAI | None = None

    # ------------------------------------------------------------------ #
    #  Client lifecycle — same pattern as OpenAIChatGenerator             #
    # ------------------------------------------------------------------ #

    def _client_kwargs(self) -> dict[str, Any]:
        """Shared config for both sync and async OpenAI clients."""
        timeout = self.timeout if self.timeout is not None else float(os.environ.get("OPENAI_TIMEOUT", "30.0"))
        max_retries = (
            self.max_retries if self.max_retries is not None else int(os.environ.get("OPENAI_MAX_RETRIES", "5"))
        )
        return {
            "api_key": self.api_key.resolve_value(),
            "organization": self.organization,
            "base_url": self.api_base_url,
            "timeout": timeout,
            "max_retries": max_retries,
        }

    def warm_up(self) -> None:
        """Initialize the synchronous OpenAI client."""
        if self.client is None:
            self.client = OpenAI(
                http_client=init_http_client(self.http_client_kwargs, async_client=False), **self._client_kwargs()
            )

    async def warm_up_async(self) -> None:  # noqa: RUF029
        """Initialize the asynchronous OpenAI client on the serving event loop."""
        if self.async_client is None:
            self.async_client = AsyncOpenAI(
                http_client=init_http_client(self.http_client_kwargs, async_client=True), **self._client_kwargs()
            )

    def close(self) -> None:
        """Release the synchronous OpenAI client."""
        if self.client is not None:
            self.client.close()
            self.client = None

    async def close_async(self) -> None:
        """Release the asynchronous OpenAI client."""
        if self.async_client is not None:
            await self.async_client.close()
            self.async_client = None

    def _get_telemetry_data(self) -> dict[str, Any]:
        """Data that is sent to Posthog for usage analytics."""
        return {"model": self.model}

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :returns:
            The serialized component as a dictionary.
        """
        return default_to_dict(
            self,
            model=self.model,
            api_base_url=self.api_base_url,
            organization=self.organization,
            generation_kwargs=self.generation_kwargs,
            api_key=self.api_key,
            timeout=self.timeout,
            max_retries=self.max_retries,
            poll_interval=self.poll_interval,
            max_wait_seconds=self.max_wait_seconds,
            completion_window=self.completion_window,
            http_client_kwargs=self.http_client_kwargs,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OpenAIBatchChatGenerator":
        """
        Deserialize this component from a dictionary.

        :param data: The dictionary representation of this component.
        :returns:
            The deserialized component instance.
        """
        return default_from_dict(cls, data)

    @component.output_types(replies=list[list[ChatMessage]], meta=dict[str, Any])
    def run(
        self, message_sets: list[list[ChatMessage]], generation_kwargs: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Submit multiple conversations to OpenAI's Batch API and wait for results.

        Each item in ``message_sets`` is a separate conversation (a list of ChatMessage objects)
        that becomes one request in the batch. The method blocks until the batch reaches a
        terminal status or the ``max_wait_seconds`` timeout is reached.

        :param message_sets:
            A list of conversations. Each conversation is a list of ChatMessage instances
            representing the full message history (system prompt, user messages, etc.).
        :param generation_kwargs:
            Runtime overrides for generation parameters. Merged with (and takes precedence over)
            the init-time ``generation_kwargs``.
        :returns:
            A dictionary with:
            - ``replies``: A list of lists, where ``replies[i]`` contains the ChatMessage
              response(s) for ``message_sets[i]``.
            - ``meta``: Batch-level metadata including ``batch_id``, ``request_counts``,
              and timestamps.
        """
        self.warm_up()
        assert self.client is not None  # mypy: guaranteed by warm_up

        if not message_sets:
            return {"replies": [], "meta": {}}

        merged_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}

        # Build the JSONL payload, upload it, and kick off the batch
        jsonl_content = _build_jsonl(message_sets, self.model, merged_kwargs)
        input_file = self.client.files.create(file=("batch_input.jsonl", jsonl_content), purpose="batch")

        batch = self.client.batches.create(
            input_file_id=input_file.id, endpoint=_BATCH_ENDPOINT, completion_window=self.completion_window
        )
        logger.info("Batch {batch_id} created with {n} requests.", batch_id=batch.id, n=len(message_sets))

        # Wait for the batch to finish
        batch = self._poll_sync(batch.id)
        _raise_on_failure(batch)

        # Download and parse the output
        assert batch.output_file_id is not None  # guaranteed when status == "completed"
        output_content = self.client.files.content(batch.output_file_id)
        replies = _parse_results(output_content.text, len(message_sets))

        return {"replies": replies, "meta": _batch_meta(batch)}

    # ------------------------------------------------------------------ #
    #  run_async() — asynchronous entry point                             #
    # ------------------------------------------------------------------ #

    @component.output_types(replies=list[list[ChatMessage]], meta=dict[str, Any])
    async def run_async(
        self, message_sets: list[list[ChatMessage]], generation_kwargs: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Async version of ``run()``. Uses ``asyncio.sleep`` for polling instead of blocking.

        See ``run()`` for full parameter documentation.
        """
        await self.warm_up_async()
        assert self.async_client is not None  # mypy: guaranteed by warm_up_async

        if not message_sets:
            return {"replies": [], "meta": {}}

        merged_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}

        jsonl_content = _build_jsonl(message_sets, self.model, merged_kwargs)
        input_file = await self.async_client.files.create(file=("batch_input.jsonl", jsonl_content), purpose="batch")

        batch = await self.async_client.batches.create(
            input_file_id=input_file.id, endpoint=_BATCH_ENDPOINT, completion_window=self.completion_window
        )
        logger.info("Batch {batch_id} created with {n} requests.", batch_id=batch.id, n=len(message_sets))

        batch = await self._poll_async(batch.id)
        _raise_on_failure(batch)

        assert batch.output_file_id is not None
        output_response = await self.async_client.files.content(batch.output_file_id)
        replies = _parse_results(output_response.text, len(message_sets))

        return {"replies": replies, "meta": _batch_meta(batch)}

    def _poll_sync(self, batch_id: str) -> Batch:
        """Poll the batch status synchronously until it reaches a terminal state."""
        start = time.monotonic()

        while True:
            batch = self.client.batches.retrieve(batch_id)  # type: ignore

            if batch.status in _TERMINAL_STATUSES:
                logger.info("Batch {batch_id} reached status '{status}'.", batch_id=batch_id, status=batch.status)
                return batch

            elapsed = time.monotonic() - start
            if elapsed >= self.max_wait_seconds:
                raise TimeoutError(
                    f"Batch {batch_id} did not complete within {self.max_wait_seconds}s. Last status: '{batch.status}'."
                )

            logger.debug(
                "Batch {batch_id} status: '{status}'. Checking again in {interval}s.",
                batch_id=batch_id,
                status=batch.status,
                interval=self.poll_interval,
            )
            time.sleep(self.poll_interval)

    async def _poll_async(self, batch_id: str) -> Batch:
        """Poll the batch status asynchronously, yielding to the event loop between checks."""
        start = time.monotonic()

        while True:
            batch = await self.async_client.batches.retrieve(batch_id)  # type: ignore

            if batch.status in _TERMINAL_STATUSES:
                logger.info("Batch {batch_id} reached status '{status}'.", batch_id=batch_id, status=batch.status)
                return batch

            elapsed = time.monotonic() - start
            if elapsed >= self.max_wait_seconds:
                raise TimeoutError(
                    f"Batch {batch_id} did not complete within {self.max_wait_seconds}s. Last status: '{batch.status}'."
                )

            logger.debug(
                "Batch {batch_id} status: '{status}'. Checking again in {interval}s.",
                batch_id=batch_id,
                status=batch.status,
                interval=self.poll_interval,
            )
            await asyncio.sleep(self.poll_interval)


def _build_jsonl(message_sets: list[list[ChatMessage]], model: str, generation_kwargs: dict[str, Any]) -> io.BytesIO:
    """
    Convert a list of conversations into a JSONL payload for the Batch API.

    Each conversation gets a ``custom_id`` like ``"request-0"``, ``"request-1"``, etc.
    so we can match results back to the original order — OpenAI doesn't guarantee
    output ordering.
    """
    lines: list[str] = []
    for idx, messages in enumerate(message_sets):
        openai_messages = [msg.to_openai_dict_format() for msg in messages]
        request_body = {"model": model, "messages": openai_messages, **generation_kwargs}
        line = {"custom_id": f"request-{idx}", "method": "POST", "url": _BATCH_ENDPOINT, "body": request_body}
        lines.append(json.dumps(line))

    return io.BytesIO("\n".join(lines).encode("utf-8"))


def _parse_results(output_text: str, expected_count: int) -> list[list[ChatMessage]]:
    """
    Parse the batch output JSONL into ChatMessage objects, ordered by request index.

    The Batch API returns results as raw JSON dicts — not pydantic ``ChatCompletion`` objects —
    so we extract the fields manually. The logic mirrors ``_convert_chat_completion_to_chat_message``
    from ``openai.py``, but works on plain dicts instead of SDK models.
    """
    # Index results by custom_id for O(1) lookup
    results_by_id: dict[str, dict[str, Any]] = {}
    for line in output_text.strip().split("\n"):
        if not line:
            continue
        entry = json.loads(line)
        results_by_id[entry["custom_id"]] = entry

    # Reassemble in the original request order
    replies: list[list[ChatMessage]] = []
    for idx in range(expected_count):
        custom_id = f"request-{idx}"
        entry = results_by_id.get(custom_id)

        if entry is None:
            # Shouldn't happen with a completed batch, but don't crash
            logger.warning("No result found for {custom_id} in batch output.", custom_id=custom_id)
            replies.append([])
            continue

        # Per-request errors (e.g., content filter, invalid model)
        if entry.get("error") is not None:
            error_info = entry["error"]
            logger.warning("Request {custom_id} failed: {error}", custom_id=custom_id, error=error_info)
            replies.append([])
            continue

        response = entry.get("response", {})
        if response.get("status_code") != 200:
            logger.warning(
                "Request {custom_id} returned HTTP {status_code}.",
                custom_id=custom_id,
                status_code=response.get("status_code"),
            )
            replies.append([])
            continue

        response_body = response.get("body", {})
        conversation_replies = _parse_choices(response_body)
        replies.append(conversation_replies)

    return replies


def _parse_choices(response_body: dict[str, Any]) -> list[ChatMessage]:
    """
    Turn the ``choices`` array from a raw completion dict into ChatMessage objects.

    This is the dict-based equivalent of ``_convert_chat_completion_to_chat_message``
    in openai.py. We can't reuse that function directly because it expects pydantic
    ``ChatCompletion``/``Choice`` objects, but batch output is plain JSON.
    """
    choices = response_body.get("choices", [])
    model = response_body.get("model", "")
    usage = response_body.get("usage")

    messages: list[ChatMessage] = []
    for choice in choices:
        message_data = choice.get("message", {})
        text = message_data.get("content")

        # Parse tool calls if present — not in MVP scope, but costs nothing
        # to handle correctly and avoids data loss if someone passes tool-enabled
        # generation_kwargs anyway
        tool_calls: list[ToolCall] = []
        for tc in message_data.get("tool_calls") or []:
            fn = tc.get("function", {})
            try:
                arguments = json.loads(fn.get("arguments", "{}"))
            except json.JSONDecodeError:
                logger.warning(
                    "Malformed JSON in tool call arguments. Tool call ID: {tc_id}, "
                    "Tool name: {tc_name}, Arguments: {tc_args}",
                    tc_id=tc.get("id"),
                    tc_name=fn.get("name"),
                    tc_args=fn.get("arguments"),
                )
                arguments = {}
            tool_calls.append(ToolCall(id=tc.get("id", ""), tool_name=fn.get("name", ""), arguments=arguments))

        meta: dict[str, Any] = {
            "model": model,
            "index": choice.get("index", 0),
            "finish_reason": choice.get("finish_reason"),
            "usage": usage,
        }

        messages.append(ChatMessage.from_assistant(text=text, tool_calls=tool_calls or None, meta=meta))

    return messages


def _raise_on_failure(batch: Batch) -> None:
    """Raise a RuntimeError if the batch didn't complete successfully."""
    if batch.status == "completed":
        return

    error_messages: list[str] = []
    if batch.errors and batch.errors.data:
        error_messages = [e.message for e in batch.errors.data if e.message]

    detail = "; ".join(error_messages) if error_messages else "No error details available."
    raise RuntimeError(f"Batch {batch.id} finished with status '{batch.status}'. {detail}")


def _batch_meta(batch: Batch) -> dict[str, Any]:
    """Extract batch-level metadata into a plain dict for the output."""
    return {
        "batch_id": batch.id,
        "status": batch.status,
        "created_at": batch.created_at,
        "completed_at": batch.completed_at,
        "request_counts": batch.request_counts.model_dump() if batch.request_counts else None,
    }
