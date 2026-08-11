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
from haystack.components.generators.chat.utils import _generator_output_token_limit_key, _run_kwargs_with_output_limit
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


class TestRunKwargsWithOutputLimit:
    def test_passes_the_limit_as_the_providers_runtime_setting(self):
        kwargs = _run_kwargs_with_output_limit(chat_generator=OpenAIChatGenerator(), limit=100)

        assert kwargs == {"generation_kwargs": {"max_completion_tokens": 100}}

    def test_overrides_a_limit_the_generator_already_configures(self):
        # A caller that reserves room for a reply of a given size needs that size honored, so the runtime value wins.
        generator = OpenAIChatGenerator(generation_kwargs={"temperature": 0, "max_completion_tokens": 23})

        kwargs = _run_kwargs_with_output_limit(chat_generator=generator, limit=100)

        assert kwargs == {"generation_kwargs": {"max_completion_tokens": 100}}
        # Only this call is affected; the generator keeps its own settings, including the ones it is not asked about.
        assert generator.generation_kwargs == {"temperature": 0, "max_completion_tokens": 23}

    def test_the_override_reaches_the_generator(self):
        # Generators merge runtime kwargs over configured ones, which is what makes the override above take effect.
        generator = OpenAIChatGenerator(generation_kwargs={"temperature": 0, "max_completion_tokens": 23})
        kwargs = _run_kwargs_with_output_limit(chat_generator=generator, limit=100)

        merged = {**generator.generation_kwargs, **kwargs["generation_kwargs"]}

        assert merged == {"temperature": 0, "max_completion_tokens": 100}

    def test_an_unknown_generator_receives_no_guessed_setting(self):
        # The protocol only guarantees `run(messages)`, so even an empty `generation_kwargs` could raise a TypeError.
        generator = integration_generator("MysteryChatGenerator")

        assert _run_kwargs_with_output_limit(chat_generator=generator, limit=100) == {}
