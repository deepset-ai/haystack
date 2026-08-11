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
from haystack.components.generators.chat.utils import _generator_output_token_limit_key, _resolve_output_token_limit
from haystack.dataclasses import ChatMessage


def integration_generator(class_name: str) -> ChatGenerator:
    """
    Stand in for a Chat Generator that lives in `haystack-core-integrations`.

    Those packages are not importable from here, which is why the lookup matches on class name. A stand-in named the
    same way is therefore an accurate test of the mechanism, but it cannot catch a rename on the integration side.
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
            ("AmazonBedrockChatGenerator", "maxTokens"),
            ("AnthropicChatGenerator", "max_tokens"),
            ("GoogleAIGeminiChatGenerator", "max_output_tokens"),
            ("GoogleGenAIChatGenerator", "max_output_tokens"),
            ("HuggingFaceAPIChatGenerator", "max_tokens"),
            ("LiteLLMChatGenerator", "max_tokens"),
            ("LlamaCppChatGenerator", "max_tokens"),
            ("OllamaChatGenerator", "num_predict"),
            ("TransformersChatGenerator", "max_new_tokens"),
            ("VertexAIGeminiChatGenerator", "max_output_tokens"),
            ("VLLMChatGenerator", "max_tokens"),
            ("WatsonxChatGenerator", "max_new_tokens"),
        ],
    )
    def test_recognizes_integration_generators(self, class_name, expected):
        assert _generator_output_token_limit_key(chat_generator=integration_generator(class_name)) == expected

    def test_a_subclass_resolves_through_the_generator_it_inherits_from(self):
        # This is what covers the OpenAI-compatible integrations, such as Mistral, OpenRouter, and TogetherAI, without
        # naming any of them.
        class ProviderChatGenerator(OpenAIChatGenerator):
            pass

        assert _generator_output_token_limit_key(chat_generator=ProviderChatGenerator()) == "max_completion_tokens"

    def test_the_most_derived_entry_wins(self):
        # No shipped generator depends on this today, since every subclass that is listed agrees with its base. It is
        # pinned so that adding an entry for a subclass cannot be silently overridden by the base it inherits from.
        base = type("OpenAIChatGenerator", (), {})
        derived = type("OpenAIResponsesChatGenerator", (base,), {})

        assert _generator_output_token_limit_key(chat_generator=derived()) == "max_output_tokens"

    def test_unknown_generator_is_not_recognized(self):
        assert _generator_output_token_limit_key(chat_generator=integration_generator("MysteryChatGenerator")) is None


class TestResolveOutputTokenLimit:
    def test_sends_the_default_as_the_providers_runtime_setting(self):
        assert _resolve_output_token_limit(chat_generator=OpenAIChatGenerator(), default_limit=100) == (
            100,
            {"max_completion_tokens": 100},
        )

    def test_a_configured_limit_wins_and_is_not_repeated_at_runtime(self):
        generator = OpenAIChatGenerator(generation_kwargs={"temperature": 0, "max_completion_tokens": 23})
        original = dict(generator.generation_kwargs)

        assert _resolve_output_token_limit(chat_generator=generator, default_limit=100) == (23, None)
        assert generator.generation_kwargs == original

    @pytest.mark.parametrize("value", [0, -1, True, "512", None], ids=["zero", "negative", "bool", "string", "none"])
    def test_an_invalid_configured_limit_is_left_for_the_generator_to_report(self, value):
        generator = OpenAIChatGenerator(generation_kwargs={"max_completion_tokens": value})
        # The generator owns the setting, so it is neither used nor silently overwritten at runtime.
        assert _resolve_output_token_limit(chat_generator=generator, default_limit=100) == (100, None)

    def test_an_unknown_generator_receives_no_guessed_setting(self):
        # The protocol only guarantees `run(messages)`, so passing a guessed kwarg would raise inside the generator.
        generator = integration_generator("MysteryChatGenerator")
        assert _resolve_output_token_limit(chat_generator=generator, default_limit=100) == (100, None)
