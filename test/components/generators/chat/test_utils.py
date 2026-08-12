# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

from haystack.components.generators.chat import (
    AzureOpenAIChatGenerator,
    AzureOpenAIResponsesChatGenerator,
    OpenAIChatGenerator,
    OpenAIResponsesChatGenerator,
)
from haystack.components.generators.chat.types import ChatGenerator
from haystack.components.generators.chat.utils import _generator_output_token_limit_key
from haystack.dataclasses import ChatMessage


def integration_generator(class_name: str) -> ChatGenerator:
    """
    Return an object named `class_name` that satisfies `ChatGenerator`.

    It stands in for a generator from `haystack-core-integrations`, which cannot be imported here. Lookup is by class
    name, so the name is the only part that has to match; a rename on the integration side goes unnoticed.
    """

    def run(self: Any, messages: list[ChatMessage], **kwargs: Any) -> dict[str, Any]:
        return {"replies": []}

    generator: ChatGenerator = type(class_name, (), {"run": run})()
    return generator


class TestGeneratorOutputTokenLimitKey:
    @pytest.mark.parametrize(
        ("generator", "expected"),
        [
            pytest.param(OpenAIChatGenerator(), "max_completion_tokens", id="openai"),
            pytest.param(
                AzureOpenAIChatGenerator(azure_endpoint="https://test.openai.azure.com"),
                "max_completion_tokens",
                id="azure-openai",
            ),
            pytest.param(OpenAIResponsesChatGenerator(), "max_output_tokens", id="openai-responses"),
            pytest.param(
                AzureOpenAIResponsesChatGenerator(azure_endpoint="https://test.openai.azure.com"),
                "max_output_tokens",
                id="azure-openai-responses",
            ),
        ],
    )
    def test_recognizes_built_in_generators(self, generator, expected):
        assert _generator_output_token_limit_key(chat_generator=generator) == expected

    @pytest.mark.parametrize(
        ("class_name", "expected"),
        [
            ("AIMLAPIChatGenerator", "max_tokens"),
            ("AmazonBedrockChatGenerator", "maxTokens"),
            ("AnthropicChatGenerator", "max_tokens"),
            ("AnthropicFoundryChatGenerator", "max_tokens"),
            ("AnthropicVertexChatGenerator", "max_tokens"),
            ("CohereChatGenerator", "max_tokens"),
            ("CometAPIChatGenerator", "max_tokens"),
            ("EdenAIChatGenerator", "max_tokens"),
            ("GoogleAIGeminiChatGenerator", "max_output_tokens"),
            ("GoogleGenAIChatGenerator", "max_output_tokens"),
            ("HuggingFaceAPIChatGenerator", "max_tokens"),
            ("LiteLLMChatGenerator", "max_tokens"),
            ("LlamaCppChatGenerator", "max_tokens"),
            ("LlamaStackChatGenerator", "max_tokens"),
            ("MistralChatGenerator", "max_tokens"),
            ("NvidiaChatGenerator", "max_tokens"),
            ("OllamaChatGenerator", "num_predict"),
            ("OpenRouterChatGenerator", "max_tokens"),
            ("OrcaRouterChatGenerator", "max_tokens"),
            ("PerplexityChatGenerator", "max_output_tokens"),
            ("STACKITChatGenerator", "max_tokens"),
            ("TogetherAIChatGenerator", "max_tokens"),
            ("TransformersChatGenerator", "max_new_tokens"),
            ("VertexAIGeminiChatGenerator", "max_output_tokens"),
            ("VLLMChatGenerator", "max_tokens"),
            ("WatsonxChatGenerator", "max_new_tokens"),
        ],
    )
    def test_recognizes_integration_generators(self, class_name, expected):
        assert _generator_output_token_limit_key(chat_generator=integration_generator(class_name)) == expected

    def test_an_unlisted_subclass_is_not_recognized(self):
        class ProviderChatGenerator(OpenAIChatGenerator):
            pass

        assert _generator_output_token_limit_key(chat_generator=ProviderChatGenerator()) is None

    def test_unknown_generator_is_not_recognized(self):
        assert _generator_output_token_limit_key(chat_generator=integration_generator("MysteryChatGenerator")) is None
