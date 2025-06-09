import os
import logging
from typing import List, Dict, Any, Optional, AsyncIterator, Union
from unittest.mock import MagicMock, patch, AsyncMock
import pytest
from haystack.utils import Secret
from haystack.components.generators.chat.watsonx import WatsonxChatGenerator
from haystack.dataclasses import ChatMessage, StreamingChunk, ToolCall

logger = logging.getLogger(__name__)


class TestWatsonxChatGenerator:
    @pytest.fixture
    def mock_watsonx(self, monkeypatch):
        """Fixture for setting up common mocks"""
        monkeypatch.setenv("WATSONX_API_KEY", "fake-api-key")

        with patch("haystack.components.generators.chat.watsonx.ModelInference") as mock_model:
            mock_model_instance = MagicMock()
            mock_model_instance.chat = MagicMock(
                return_value={
                    "choices": [
                        {
                            "message": {"content": "This is a generated response", "role": "assistant"},
                            "finish_reason": "completed",
                        }
                    ]
                }
            )
            mock_model_instance.achat = AsyncMock(
                return_value={
                    "choices": [
                        {
                            "message": {"content": "Async generated response", "role": "assistant"},
                            "finish_reason": "completed",
                        }
                    ]
                }
            )
            mock_model_instance.chat_stream = MagicMock(
                return_value=[
                    {"choices": [{"delta": {"content": "Streaming"}, "finish_reason": None}]},
                    {"choices": [{"delta": {"content": " response"}, "finish_reason": "completed"}]},
                ]
            )

            async def mock_achat_stream_generator():
                yield {"choices": [{"delta": {"content": "Async streaming"}, "finish_reason": None}]}
                yield {"choices": [{"delta": {"content": " response"}, "finish_reason": "completed"}]}

            async def mock_achat_stream(messages=None, params=None):
                return mock_achat_stream_generator()

            mock_model_instance.achat_stream = mock_achat_stream
            mock_model.return_value = mock_model_instance

            yield {"model": mock_model, "model_instance": mock_model_instance}

    def test_init_default(self, mock_watsonx):
        generator = WatsonxChatGenerator(model="ibm/granite-3-2b-instruct", project_id="fake-project-id")

        args, kwargs = mock_watsonx["model"].call_args
        assert kwargs["model_id"] == "ibm/granite-3-2b-instruct"
        assert kwargs["project_id"] == "fake-project-id"
        assert kwargs["space_id"] is None
        assert kwargs["verify"] is None

        assert generator.model == "ibm/granite-3-2b-instruct"
        assert generator.project_id == "fake-project-id"
        assert generator.space_id is None
        assert generator.api_base_url == "https://us-south.ml.cloud.ibm.com"

    def test_init_with_all_params(self, mock_watsonx):
        generator = WatsonxChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            model="ibm/granite-3-2b-instruct",
            project_id="test-project",
            space_id="test-space",
            api_base_url="https://custom-url.com",
            generation_kwargs={"max_new_tokens": 100, "temperature": 0.7, "top_p": 0.9},
            verify=False,
        )

        args, kwargs = mock_watsonx["model"].call_args
        assert kwargs["model_id"] == "ibm/granite-3-2b-instruct"
        assert kwargs["project_id"] == "test-project"
        assert kwargs["space_id"] == "test-space"
        assert kwargs["verify"] is False

    def test_init_fails_without_project_or_space(self, mock_watsonx):
        with pytest.raises(ValueError, match="Either project_id or space_id must be provided"):
            WatsonxChatGenerator(api_key=Secret.from_token("test-api-key"), model="ibm/granite-3-2b-instruct")

    def test_to_dict(self, mock_watsonx):
        generator = WatsonxChatGenerator(
            model="ibm/granite-3-2b-instruct", project_id="test-project", generation_kwargs={"max_new_tokens": 100}
        )

        data = generator.to_dict()

        expected = {
            "type": "haystack.components.generators.chat.watsonx.WatsonxChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["WATSONX_API_KEY"], "strict": True, "type": "env_var"},
                "model": "ibm/granite-3-2b-instruct",
                "project_id": "test-project",
                "space_id": None,
                "api_base_url": "https://us-south.ml.cloud.ibm.com",
                "generation_kwargs": {"max_new_tokens": 100},
                "verify": None,
                "timeout": 30.0,
                "max_retries": 5,
                "tools": None,
            },
        }
        assert data == expected

    def test_from_dict(self, mock_watsonx):
        data = {
            "type": "haystack.components.generators.chat.watsonx.WatsonxChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["WATSONX_API_KEY"], "strict": True, "type": "env_var"},
                "model": "ibm/granite-3-2b-instruct",
                "project_id": "test-project",
                "generation_kwargs": {"max_new_tokens": 100},
            },
        }

        generator = WatsonxChatGenerator.from_dict(data)
        assert generator.model == "ibm/granite-3-2b-instruct"
        assert generator.project_id == "test-project"
        assert generator.generation_kwargs == {"max_new_tokens": 100}

    def test_run_single_message(self, mock_watsonx):
        generator = WatsonxChatGenerator(
            api_key=Secret.from_token("test-api-key"), model="ibm/granite-3-2b-instruct", project_id="test-project"
        )

        messages = [ChatMessage.from_user("Test prompt")]
        result = generator.run(messages)

        assert len(result["replies"]) == 1
        assert result["replies"][0].text == "This is a generated response"
        assert result["replies"][0].meta["finish_reason"] == "completed"

        mock_watsonx["model_instance"].chat.assert_called_once_with(
            messages=[{"role": "user", "content": "Test prompt"}], params={}
        )

    def test_run_with_generation_params(self, mock_watsonx):
        generator = WatsonxChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            model="ibm/granite-3-2b-instruct",
            project_id="test-project",
            generation_kwargs={"max_new_tokens": 100, "temperature": 0.7, "top_p": 0.9},
        )

        messages = [ChatMessage.from_user("Test prompt")]
        result = generator.run(messages)

        assert len(result["replies"]) == 1
        mock_watsonx["model_instance"].chat.assert_called_once_with(
            messages=[{"role": "user", "content": "Test prompt"}],
            params={"max_new_tokens": 100, "temperature": 0.7, "top_p": 0.9},
        )

    def test_run_with_streaming(self, mock_watsonx):
        generator = WatsonxChatGenerator(
            api_key=Secret.from_token("test-api-key"), model="ibm/granite-3-2b-instruct", project_id="test-project"
        )

        messages = [ChatMessage.from_user("Test prompt")]
        result = generator.run(messages, stream=True)

        assert len(result["replies"]) == 1
        assert result["replies"][0].text == "Streaming response"
        assert len(result["chunks"]) == 2
        assert isinstance(result["chunks"][0], StreamingChunk)
        assert result["chunks"][0].content == "Streaming"

    def test_run_with_empty_messages(self, mock_watsonx):
        generator = WatsonxChatGenerator(
            api_key=Secret.from_token("test-api-key"), model="ibm/granite-3-2b-instruct", project_id="test-project"
        )

        result = generator.run([])
        assert result["replies"] == []

    def test_run_with_tool_calls(self, mock_watsonx):
        mock_watsonx["model_instance"].chat.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": '{"location": "Berlin"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }

        generator = WatsonxChatGenerator(
            api_key=Secret.from_token("test-api-key"), model="ibm/granite-3-2b-instruct", project_id="test-project"
        )

        messages = [ChatMessage.from_user("What's the weather in Berlin?")]
        result = generator.run(messages)

        assert len(result["replies"]) == 1
        assert len(result["replies"][0].tool_calls) == 1
        assert result["replies"][0].tool_calls[0].tool_name == "get_weather"
        assert result["replies"][0].tool_calls[0].arguments == {"location": "Berlin"}

    @pytest.mark.asyncio
    async def test_run_async_single_message(self, mock_watsonx):
        generator = WatsonxChatGenerator(
            api_key=Secret.from_token("test-api-key"), model="ibm/granite-3-2b-instruct", project_id="test-project"
        )

        messages = [ChatMessage.from_user("Test prompt")]
        result = await generator.run_async(messages)

        assert len(result["replies"]) == 1
        assert result["replies"][0].text == "Async generated response"
        assert result["replies"][0].meta["finish_reason"] == "completed"

    @pytest.mark.asyncio
    async def test_run_async_streaming(self, mock_watsonx):
        generator = WatsonxChatGenerator(
            api_key=Secret.from_token("test-api-key"), model="ibm/granite-3-2b-instruct", project_id="test-project"
        )

        messages = [ChatMessage.from_user("Test prompt")]
        result = await generator.run_async(messages, stream=True)

        assert len(result["replies"]) == 1
        assert result["replies"][0].text == "Async streaming response"
        assert len(result["chunks"]) == 2
        assert isinstance(result["chunks"][0], StreamingChunk)
        assert result["chunks"][0].content == "Async streaming"


@pytest.mark.integration
class TestWatsonxChatGeneratorIntegration:
    @pytest.mark.skipif(
        not os.environ.get("WATSONX_API_KEY") or not os.environ.get("WATSONX_PROJECT_ID"),
        reason="WATSONX_API_KEY or WATSONX_PROJECT_ID not set",
    )
    def test_live_run(self):
        generator = WatsonxChatGenerator(
            model="ibm/granite-3-2b-instruct",
            project_id=os.environ["WATSONX_PROJECT_ID"],
            generation_kwargs={"max_new_tokens": 50, "temperature": 0.7, "top_p": 0.9},
        )
        messages = [ChatMessage.from_user("What's the capital of France?")]
        results = generator.run(messages)

        assert isinstance(results, dict)
        assert "replies" in results
        assert isinstance(results["replies"], list)
        assert len(results["replies"]) == 1
        assert isinstance(results["replies"][0], ChatMessage)
        assert len(results["replies"][0].text) > 0

    @pytest.mark.skipif(
        not os.environ.get("WATSONX_API_KEY") or not os.environ.get("WATSONX_PROJECT_ID"),
        reason="WATSONX_API_KEY or WATSONX_PROJECT_ID not set",
    )
    def test_live_streaming(self):
        generator = WatsonxChatGenerator(model="ibm/granite-3-2b-instruct", project_id=os.environ["WATSONX_PROJECT_ID"])

        messages = [ChatMessage.from_user("Explain quantum computing")]
        results = generator.run(messages, stream=True)

        assert isinstance(results, dict)
        assert "replies" in results
        assert "chunks" in results
        assert len(results["replies"]) == 1
        assert isinstance(results["replies"][0], ChatMessage)
        assert all(isinstance(chunk, StreamingChunk) for chunk in results["chunks"])

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.environ.get("WATSONX_API_KEY") or not os.environ.get("WATSONX_PROJECT_ID"),
        reason="WATSONX_API_KEY or WATSONX_PROJECT_ID not set",
    )
    async def test_live_async(self):
        generator = WatsonxChatGenerator(model="ibm/granite-3-2b-instruct", project_id=os.environ["WATSONX_PROJECT_ID"])
        messages = [ChatMessage.from_user("What's the capital of Germany?")]
        results = await generator.run_async(messages)

        assert isinstance(results, dict)
        assert "replies" in results
        assert isinstance(results["replies"], list)
        assert len(results["replies"]) == 1
        assert isinstance(results["replies"][0], ChatMessage)
        assert len(results["replies"][0].text) > 0
