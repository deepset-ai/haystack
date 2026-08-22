# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Literal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai.types import Batch, BatchError, BatchRequestCounts
from openai.types.batch import Errors

from haystack.components.generators.chat.openai_batch import (
    OpenAIBatchChatGenerator,
    _batch_meta,
    _build_jsonl,
    _parse_choices,
    _parse_results,
    _raise_on_failure,
)
from haystack.dataclasses import ChatMessage
from haystack.utils.auth import Secret


@pytest.fixture
def component():
    """A default-configured component for testing."""
    return OpenAIBatchChatGenerator(api_key=Secret.from_token("test-api-key"))


@pytest.fixture
def custom_component():
    """A fully-customized component to verify all init params are stored."""
    return OpenAIBatchChatGenerator(
        api_key=Secret.from_token("custom-key"),
        model="gpt-5",
        api_base_url="https://custom.openai.com",
        organization="org-123",
        generation_kwargs={"temperature": 0.7, "max_completion_tokens": 500},
        timeout=60.0,
        max_retries=3,
        poll_interval=10.0,
        max_wait_seconds=3600.0,
        completion_window="24h",
        http_client_kwargs={"verify": False},
    )


@pytest.fixture
def single_conversation():
    """One conversation with a system prompt and a user message."""
    return [
        ChatMessage.from_system("You are a helpful assistant."),
        ChatMessage.from_user("What is the capital of France?"),
    ]


@pytest.fixture
def two_conversations(single_conversation):
    """Two separate conversations for batch processing."""
    return [single_conversation, [ChatMessage.from_user("What is 2+2?")]]


BatchStatus = Literal[
    "validating", "failed", "in_progress", "finalizing", "completed", "expired", "cancelling", "cancelled"
]


def _make_batch(
    batch_id: str = "batch_abc",
    status: BatchStatus = "completed",
    output_file_id: str | None = "file-out-123",
    error_file_id: str | None = None,
    errors: Errors | None = None,
) -> Batch:
    """Build a Batch object for mocking, without hitting the API."""
    return Batch(
        id=batch_id,
        object="batch",
        endpoint="/v1/chat/completions",
        input_file_id="file-in-456",
        completion_window="24h",
        status=status,
        output_file_id=output_file_id,
        error_file_id=error_file_id,
        errors=errors,
        created_at=1700000000,
        completed_at=1700001000 if status == "completed" else None,
        request_counts=BatchRequestCounts(completed=2, failed=0, total=2),
    )


def _make_output_jsonl(*responses: dict) -> str:
    """
    Build a batch output JSONL string from a list of response body dicts.

    Each dict should look like a chat completion body:
    {"choices": [{"message": {"content": "Paris"}, ...}], "model": "gpt-5-mini", ...}
    """
    lines = []
    for idx, body in enumerate(responses):
        entry = {
            "id": f"batch_req_{idx}",
            "custom_id": f"request-{idx}",
            "response": {"status_code": 200, "request_id": f"req_{idx}", "body": body},
            "error": None,
        }
        lines.append(json.dumps(entry))
    return "\n".join(lines)


def _simple_completion_body(text: str, model: str = "gpt-5-mini") -> dict:
    """A minimal chat completion body dict for testing."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1700000500,
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


class TestOpenAIBatchChatGeneratorInit:
    def test_default_params(self, component):
        assert component.model == "gpt-5-mini"
        assert component.generation_kwargs == {}
        assert component.poll_interval == 30.0
        assert component.max_wait_seconds == 86400.0
        assert component.completion_window == "24h"
        assert component.api_base_url is None
        assert component.organization is None
        assert component.timeout is None
        assert component.max_retries is None
        assert component.http_client_kwargs is None
        assert component.client is None
        assert component.async_client is None

    def test_custom_params(self, custom_component):
        assert custom_component.model == "gpt-5"
        assert custom_component.api_base_url == "https://custom.openai.com"
        assert custom_component.organization == "org-123"
        assert custom_component.generation_kwargs == {"temperature": 0.7, "max_completion_tokens": 500}
        assert custom_component.timeout == 60.0
        assert custom_component.max_retries == 3
        assert custom_component.poll_interval == 10.0
        assert custom_component.max_wait_seconds == 3600.0
        assert custom_component.http_client_kwargs == {"verify": False}


class TestOpenAIBatchChatGeneratorSerialization:
    def test_to_dict(self):
        # Token-based secrets can't be serialized (by design), so use env var
        gen = OpenAIBatchChatGenerator(api_key=Secret.from_env_var("OPENAI_API_KEY"))
        result = gen.to_dict()
        assert result["type"] == "haystack.components.generators.chat.openai_batch.OpenAIBatchChatGenerator"
        params = result["init_parameters"]
        assert params["model"] == "gpt-5-mini"
        assert params["poll_interval"] == 30.0
        assert params["max_wait_seconds"] == 86400.0
        assert params["completion_window"] == "24h"
        assert params["generation_kwargs"] == {}
        assert params["api_base_url"] is None

    def test_to_dict_custom(self):
        gen = OpenAIBatchChatGenerator(
            api_key=Secret.from_env_var("MY_KEY"),
            model="gpt-5",
            api_base_url="https://custom.openai.com",
            organization="org-123",
            generation_kwargs={"temperature": 0.7, "max_completion_tokens": 500},
            timeout=60.0,
            max_retries=3,
            poll_interval=10.0,
            max_wait_seconds=3600.0,
        )
        result = gen.to_dict()
        params = result["init_parameters"]
        assert params["model"] == "gpt-5"
        assert params["api_base_url"] == "https://custom.openai.com"
        assert params["organization"] == "org-123"
        assert params["generation_kwargs"] == {"temperature": 0.7, "max_completion_tokens": 500}
        assert params["timeout"] == 60.0
        assert params["max_retries"] == 3
        assert params["poll_interval"] == 10.0
        assert params["max_wait_seconds"] == 3600.0

    def test_from_dict_round_trip(self):
        original = OpenAIBatchChatGenerator(api_key=Secret.from_env_var("OPENAI_API_KEY"))
        serialized = original.to_dict()
        restored = OpenAIBatchChatGenerator.from_dict(serialized)
        assert restored.model == original.model
        assert restored.poll_interval == original.poll_interval
        assert restored.max_wait_seconds == original.max_wait_seconds
        assert restored.completion_window == original.completion_window
        assert restored.generation_kwargs == original.generation_kwargs


class TestOpenAIBatchChatGeneratorLifecycle:
    def test_warm_up_creates_client(self, component):
        assert component.client is None
        with patch("haystack.components.generators.chat.openai_batch.OpenAI"):
            component.warm_up()
        assert component.client is not None

    def test_warm_up_idempotent(self, component):
        with patch("haystack.components.generators.chat.openai_batch.OpenAI") as mock_cls:
            component.warm_up()
            first_client = component.client
            component.warm_up()
            # Should not create a second client
            assert component.client is first_client
            mock_cls.assert_called_once()

    def test_close_releases_client(self, component):
        with patch("haystack.components.generators.chat.openai_batch.OpenAI"):
            component.warm_up()
        assert component.client is not None
        component.close()
        assert component.client is None

    def test_close_when_no_client(self, component):
        # Should not raise
        component.close()

    @pytest.mark.asyncio
    async def test_close_async_releases_client(self, component):
        with patch("haystack.components.generators.chat.openai_batch.AsyncOpenAI"):
            await component.warm_up_async()
        assert component.async_client is not None

        # Ensure we mock the close method as AsyncMock
        component.async_client.close = AsyncMock()
        await component.close_async()
        assert component.async_client is None

    @pytest.mark.asyncio
    async def test_close_async_when_no_client(self, component):
        # Should not raise
        await component.close_async()

    def test_get_telemetry_data(self, component):
        assert component._get_telemetry_data() == {"model": "gpt-5-mini"}


class TestBuildJsonl:
    def test_single_conversation(self, single_conversation):
        result = _build_jsonl([single_conversation], "gpt-5-mini", {})
        content = result.read().decode("utf-8")
        lines = content.strip().split("\n")
        assert len(lines) == 1

        parsed = json.loads(lines[0])
        assert parsed["custom_id"] == "request-0"
        assert parsed["method"] == "POST"
        assert parsed["url"] == "/v1/chat/completions"
        assert parsed["body"]["model"] == "gpt-5-mini"
        assert len(parsed["body"]["messages"]) == 2

    def test_multiple_conversations(self, two_conversations):
        result = _build_jsonl(two_conversations, "gpt-5", {"temperature": 0.5})
        content = result.read().decode("utf-8")
        lines = content.strip().split("\n")
        assert len(lines) == 2

        # First conversation
        line0 = json.loads(lines[0])
        assert line0["custom_id"] == "request-0"
        assert line0["body"]["model"] == "gpt-5"
        assert line0["body"]["temperature"] == 0.5
        assert len(line0["body"]["messages"]) == 2

        # Second conversation
        line1 = json.loads(lines[1])
        assert line1["custom_id"] == "request-1"
        assert len(line1["body"]["messages"]) == 1

    def test_generation_kwargs_included(self, single_conversation):
        kwargs = {"temperature": 0.3, "max_completion_tokens": 100}
        result = _build_jsonl([single_conversation], "gpt-5-mini", kwargs)
        parsed = json.loads(result.read().decode("utf-8"))
        assert parsed["body"]["temperature"] == 0.3
        assert parsed["body"]["max_completion_tokens"] == 100


class TestParseResults:
    def test_single_result(self):
        output = _make_output_jsonl(_simple_completion_body("Paris"))
        replies = _parse_results(output, expected_count=1)

        assert len(replies) == 1
        assert len(replies[0]) == 1
        assert replies[0][0].text == "Paris"
        assert replies[0][0].meta["model"] == "gpt-5-mini"
        assert replies[0][0].meta["finish_reason"] == "stop"
        assert replies[0][0].meta["usage"]["total_tokens"] == 15

    def test_multiple_results(self):
        output = _make_output_jsonl(_simple_completion_body("Paris"), _simple_completion_body("4"))
        replies = _parse_results(output, expected_count=2)

        assert len(replies) == 2
        assert replies[0][0].text == "Paris"
        assert replies[1][0].text == "4"

    def test_results_reordered(self):
        """The batch output might not be in the same order as the input."""
        line0 = {
            "id": "req_1",
            "custom_id": "request-1",
            "response": {"status_code": 200, "request_id": "r1", "body": _simple_completion_body("second")},
            "error": None,
        }
        line1 = {
            "id": "req_0",
            "custom_id": "request-0",
            "response": {"status_code": 200, "request_id": "r0", "body": _simple_completion_body("first")},
            "error": None,
        }
        output = json.dumps(line0) + "\n" + json.dumps(line1)
        replies = _parse_results(output, expected_count=2)

        # Should be re-sorted by custom_id index
        assert replies[0][0].text == "first"
        assert replies[1][0].text == "second"

    def test_missing_result_returns_empty(self):
        """If a result is missing for a custom_id, we get an empty list — not a crash."""
        output = _make_output_jsonl(_simple_completion_body("Paris"))
        replies = _parse_results(output, expected_count=2)

        assert len(replies) == 2
        assert replies[0][0].text == "Paris"
        assert replies[1] == []  # missing request-1

    def test_per_request_error(self):
        """Per-request errors should produce empty replies, not crash."""
        error_line = {
            "id": "req_0",
            "custom_id": "request-0",
            "response": None,
            "error": {"code": "content_filter", "message": "Content was blocked."},
        }
        output = json.dumps(error_line)
        replies = _parse_results(output, expected_count=1)

        assert len(replies) == 1
        assert replies[0] == []

    def test_results_with_empty_lines(self):
        entry1 = {"custom_id": "request-0", "response": {"status_code": 200, "body": _simple_completion_body("A")}}
        entry2 = {"custom_id": "request-1", "response": {"status_code": 200, "body": _simple_completion_body("B")}}

        # Put an empty line right in the middle
        output = json.dumps(entry1) + "\n\n\n" + json.dumps(entry2)

        replies = _parse_results(output, expected_count=2)
        assert len(replies) == 2
        assert replies[0][0].text == "A"
        assert replies[1][0].text == "B"

    def test_per_request_http_error(self):
        # A valid JSONL entry but with an HTTP non-200 status_code in the 'response'
        entry = {"custom_id": "request-0", "response": {"status_code": 500, "body": {}}}
        output = json.dumps(entry) + "\n"
        replies = _parse_results(output, expected_count=1)
        assert len(replies) == 1
        assert replies[0] == []


class TestParseChoices:
    def test_basic_text(self):
        body = _simple_completion_body("Hello!")
        messages = _parse_choices(body)

        assert len(messages) == 1
        assert messages[0].text == "Hello!"
        assert messages[0].meta["finish_reason"] == "stop"

    def test_with_tool_calls(self):
        """Tool calls in the response should be parsed even though tools aren't in MVP scope."""
        body = {
            "model": "gpt-5-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
        }
        messages = _parse_choices(body)

        assert len(messages) == 1
        assert messages[0].tool_calls is not None
        assert len(messages[0].tool_calls) == 1
        assert messages[0].tool_calls[0].tool_name == "get_weather"
        assert messages[0].tool_calls[0].arguments == {"city": "Paris"}

    def test_multiple_choices(self):
        """When n>1, the API returns multiple choices per request."""
        body = {
            "model": "gpt-5-mini",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "Answer A"}, "finish_reason": "stop"},
                {"index": 1, "message": {"role": "assistant", "content": "Answer B"}, "finish_reason": "stop"},
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        }
        messages = _parse_choices(body)
        assert len(messages) == 2
        assert messages[0].text == "Answer A"
        assert messages[1].text == "Answer B"

    def test_malformed_tool_call_json(self):
        body = {
            "model": "gpt-5-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": '{"city": "Paris"'},  # Missing brace
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
        }
        messages = _parse_choices(body)
        assert len(messages) == 1
        assert messages[0].tool_calls is not None
        assert len(messages[0].tool_calls) == 1
        assert messages[0].tool_calls[0].arguments == {}


class TestRaiseOnFailure:
    def test_completed_does_not_raise(self):
        batch = _make_batch(status="completed")
        _raise_on_failure(batch)  # should not raise

    def test_failed_raises(self):
        batch = _make_batch(status="failed")
        with pytest.raises(RuntimeError, match="finished with status 'failed'"):
            _raise_on_failure(batch)

    def test_expired_raises(self):
        batch = _make_batch(status="expired")
        with pytest.raises(RuntimeError, match="finished with status 'expired'"):
            _raise_on_failure(batch)

    def test_error_details_included(self):
        errors = Errors(
            object="list", data=[BatchError(code="invalid_model", message="Model not found", line=1, param="model")]
        )
        batch = _make_batch(status="failed", errors=errors)
        with pytest.raises(RuntimeError, match="Model not found"):
            _raise_on_failure(batch)


class TestBatchMeta:
    def test_extracts_metadata(self):
        batch = _make_batch()
        meta = _batch_meta(batch)
        assert meta["batch_id"] == "batch_abc"
        assert meta["status"] == "completed"
        assert meta["created_at"] == 1700000000
        assert meta["completed_at"] == 1700001000
        assert meta["request_counts"]["total"] == 2


class TestOpenAIBatchChatGeneratorRun:
    def test_run_single_conversation(self, component, single_conversation):
        output_jsonl = _make_output_jsonl(_simple_completion_body("Paris"))

        mock_client = MagicMock()
        mock_client.files.create.return_value = MagicMock(id="file-in-123")
        mock_client.batches.create.return_value = _make_batch(status="validating")
        mock_client.batches.retrieve.return_value = _make_batch(status="completed")
        mock_client.files.content.return_value = MagicMock(text=output_jsonl)

        with patch("haystack.components.generators.chat.openai_batch.OpenAI", return_value=mock_client):
            component.warm_up()

        result = component.run(message_sets=[single_conversation])

        assert len(result["replies"]) == 1
        assert result["replies"][0][0].text == "Paris"
        assert result["meta"]["batch_id"] == "batch_abc"
        assert result["meta"]["status"] == "completed"

    def test_run_multiple_conversations(self, component, two_conversations):
        output_jsonl = _make_output_jsonl(_simple_completion_body("Paris"), _simple_completion_body("4"))

        mock_client = MagicMock()
        mock_client.files.create.return_value = MagicMock(id="file-in-123")
        mock_client.batches.create.return_value = _make_batch(status="validating")
        mock_client.batches.retrieve.return_value = _make_batch(status="completed")
        mock_client.files.content.return_value = MagicMock(text=output_jsonl)

        with patch("haystack.components.generators.chat.openai_batch.OpenAI", return_value=mock_client):
            component.warm_up()

        result = component.run(message_sets=two_conversations)

        assert len(result["replies"]) == 2
        assert result["replies"][0][0].text == "Paris"
        assert result["replies"][1][0].text == "4"

    def test_run_empty_input(self, component):
        result = component.run(message_sets=[])
        assert result == {"replies": [], "meta": {}}

    def test_run_merges_generation_kwargs(self, component, single_conversation):
        """Runtime generation_kwargs should merge with and override init kwargs."""
        component.generation_kwargs = {"temperature": 0.5, "max_completion_tokens": 100}
        output_jsonl = _make_output_jsonl(_simple_completion_body("test"))

        mock_client = MagicMock()
        mock_client.files.create.return_value = MagicMock(id="file-in-123")
        mock_client.batches.create.return_value = _make_batch(status="completed")
        mock_client.batches.retrieve.return_value = _make_batch(status="completed")
        mock_client.files.content.return_value = MagicMock(text=output_jsonl)

        with patch("haystack.components.generators.chat.openai_batch.OpenAI", return_value=mock_client):
            component.warm_up()

        # Runtime kwarg overrides temperature but keeps max_completion_tokens
        component.run(message_sets=[single_conversation], generation_kwargs={"temperature": 0.9})

        # Inspect the JSONL that was uploaded
        upload_call = mock_client.files.create.call_args
        file_tuple = upload_call.kwargs.get("file") or upload_call[1].get("file")
        jsonl_content = file_tuple[1].read().decode("utf-8")
        parsed = json.loads(jsonl_content)

        assert parsed["body"]["temperature"] == 0.9  # overridden
        assert parsed["body"]["max_completion_tokens"] == 100  # preserved from init

    def test_run_batch_failed_raises(self, component, single_conversation):
        mock_client = MagicMock()
        mock_client.files.create.return_value = MagicMock(id="file-in-123")
        mock_client.batches.create.return_value = _make_batch(status="validating")
        mock_client.batches.retrieve.return_value = _make_batch(status="failed")

        with patch("haystack.components.generators.chat.openai_batch.OpenAI", return_value=mock_client):
            component.warm_up()

        with pytest.raises(RuntimeError, match="finished with status 'failed'"):
            component.run(message_sets=[single_conversation])

    def test_run_batch_expired_raises(self, component, single_conversation):
        mock_client = MagicMock()
        mock_client.files.create.return_value = MagicMock(id="file-in-123")
        mock_client.batches.create.return_value = _make_batch(status="validating")
        mock_client.batches.retrieve.return_value = _make_batch(status="expired")

        with patch("haystack.components.generators.chat.openai_batch.OpenAI", return_value=mock_client):
            component.warm_up()

        with pytest.raises(RuntimeError, match="finished with status 'expired'"):
            component.run(message_sets=[single_conversation])

    def test_run_timeout_raises(self, component, single_conversation):
        """When max_wait_seconds is exceeded, a TimeoutError should be raised."""
        component.max_wait_seconds = 0.0  # Immediately times out
        component.poll_interval = 0.01

        mock_client = MagicMock()
        mock_client.files.create.return_value = MagicMock(id="file-in-123")
        mock_client.batches.create.return_value = _make_batch(status="in_progress")
        # Never reaches a terminal status
        mock_client.batches.retrieve.return_value = _make_batch(status="in_progress")

        with patch("haystack.components.generators.chat.openai_batch.OpenAI", return_value=mock_client):
            component.warm_up()

        with pytest.raises(TimeoutError, match="did not complete within"):
            component.run(message_sets=[single_conversation])

    def test_run_polls_until_completed(self, component, single_conversation):
        """Verify polling continues through non-terminal statuses."""
        component.poll_interval = 0.01  # Speed up tests

        output_jsonl = _make_output_jsonl(_simple_completion_body("result"))

        mock_client = MagicMock()
        mock_client.files.create.return_value = MagicMock(id="file-in-123")
        mock_client.batches.create.return_value = _make_batch(status="validating")
        # Simulate: validating → in_progress → finalizing → completed
        mock_client.batches.retrieve.side_effect = [
            _make_batch(status="validating"),
            _make_batch(status="in_progress"),
            _make_batch(status="finalizing"),
            _make_batch(status="completed"),
        ]
        mock_client.files.content.return_value = MagicMock(text=output_jsonl)

        with patch("haystack.components.generators.chat.openai_batch.OpenAI", return_value=mock_client):
            component.warm_up()

        result = component.run(message_sets=[single_conversation])
        assert result["replies"][0][0].text == "result"
        assert mock_client.batches.retrieve.call_count == 4


class TestOpenAIBatchChatGeneratorRunAsync:
    @pytest.mark.asyncio
    async def test_run_async_basic(self, component, single_conversation):
        output_jsonl = _make_output_jsonl(_simple_completion_body("Paris"))

        mock_client = AsyncMock()
        mock_client.files.create.return_value = MagicMock(id="file-in-123")
        mock_client.batches.create.return_value = _make_batch(status="validating")
        mock_client.batches.retrieve.return_value = _make_batch(status="completed")
        mock_client.files.content.return_value = MagicMock(text=output_jsonl)

        with patch("haystack.components.generators.chat.openai_batch.AsyncOpenAI", return_value=mock_client):
            await component.warm_up_async()

        result = await component.run_async(message_sets=[single_conversation])

        assert len(result["replies"]) == 1
        assert result["replies"][0][0].text == "Paris"
        assert result["meta"]["batch_id"] == "batch_abc"

    @pytest.mark.asyncio
    async def test_run_async_empty_input(self, component):
        result = await component.run_async(message_sets=[])
        assert result == {"replies": [], "meta": {}}

    @pytest.mark.asyncio
    async def test_run_async_batch_failed_raises(self, component, single_conversation):
        mock_client = AsyncMock()
        mock_client.files.create.return_value = MagicMock(id="file-in-123")
        mock_client.batches.create.return_value = _make_batch(status="validating")
        mock_client.batches.retrieve.return_value = _make_batch(status="failed")

        with patch("haystack.components.generators.chat.openai_batch.AsyncOpenAI", return_value=mock_client):
            await component.warm_up_async()

        with pytest.raises(RuntimeError, match="finished with status 'failed'"):
            await component.run_async(message_sets=[single_conversation])

    @pytest.mark.asyncio
    async def test_run_async_timeout_raises(self, component, single_conversation):
        component.max_wait_seconds = 0.1
        component.poll_interval = 0.5  # larger than timeout so it fails on first sleep check

        mock_client = AsyncMock()
        mock_client.files.create.return_value = MagicMock(id="file-in-123")
        mock_client.batches.create.return_value = _make_batch(status="validating")
        mock_client.batches.retrieve.return_value = _make_batch(status="in_progress")

        with patch("haystack.components.generators.chat.openai_batch.AsyncOpenAI", return_value=mock_client):
            await component.warm_up_async()

        with pytest.raises(TimeoutError, match="did not complete within"):
            await component.run_async(message_sets=[single_conversation])

    @pytest.mark.asyncio
    @patch("asyncio.sleep")
    async def test_run_async_polls_until_completed(self, mock_sleep, component, single_conversation):
        output_jsonl = _make_output_jsonl(_simple_completion_body("result"))

        mock_client = AsyncMock()
        mock_client.files.create.return_value = MagicMock(id="file-in-123")
        mock_client.batches.create.return_value = _make_batch(status="validating")

        # Simulate: validating -> in_progress -> completed
        mock_client.batches.retrieve.side_effect = [
            _make_batch(status="validating"),
            _make_batch(status="in_progress"),
            _make_batch(status="completed"),
        ]
        mock_client.files.content.return_value = MagicMock(text=output_jsonl)

        with patch("haystack.components.generators.chat.openai_batch.AsyncOpenAI", return_value=mock_client):
            await component.warm_up_async()

        result = await component.run_async(message_sets=[single_conversation])

        assert result["replies"][0][0].text == "result"
        assert mock_client.batches.retrieve.call_count == 3
        assert mock_sleep.call_count == 2
