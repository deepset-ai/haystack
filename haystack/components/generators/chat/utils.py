# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack.components.generators.chat.types import ChatGenerator

# The `generation_kwargs` key that caps a reply's length, per Chat Generator.
#
# Providers do not agree on a name for this setting and the `ChatGenerator` protocol does not standardize it, so a
# caller that wants to bound a reply has to know which key the generator expects. Generators are matched by class name
# rather than by `isinstance`, because most of the entries below live in `haystack-core-integrations` and are not
# importable from here. Only the most derived match is used, so an unlisted subclass resolves through the generator it
# inherits from: that is what covers the many OpenAI-compatible integrations without naming any of them.
_OUTPUT_TOKEN_LIMIT_KEYS = {
    # Haystack
    "OpenAIChatGenerator": "max_completion_tokens",
    "AzureOpenAIChatGenerator": "max_completion_tokens",
    "OpenAIResponsesChatGenerator": "max_output_tokens",
    "AzureOpenAIResponsesChatGenerator": "max_output_tokens",
    # haystack-core-integrations
    "AmazonBedrockChatGenerator": "maxTokens",
    "AnthropicChatGenerator": "max_tokens",
    "GoogleAIGeminiChatGenerator": "max_output_tokens",
    "GoogleGenAIChatGenerator": "max_output_tokens",
    "HuggingFaceAPIChatGenerator": "max_tokens",
    "LiteLLMChatGenerator": "max_tokens",
    "LlamaCppChatGenerator": "max_tokens",
    "OllamaChatGenerator": "num_predict",
    "TransformersChatGenerator": "max_new_tokens",
    "VertexAIGeminiChatGenerator": "max_output_tokens",
    "VLLMChatGenerator": "max_tokens",
    "WatsonxChatGenerator": "max_new_tokens",
}


def _generator_output_token_limit_key(chat_generator: ChatGenerator) -> str | None:
    """
    Return the `generation_kwargs` key that limits a Chat Generator's output length.

    :param chat_generator: The generator to look up.
    :returns: The key the generator expects, or None when the generator is not recognized.
    """
    # Walk from the most derived class, so a provider's own entry wins over the generator it inherits from.
    for cls in type(chat_generator).__mro__:
        key = _OUTPUT_TOKEN_LIMIT_KEYS.get(cls.__name__)
        if key is not None:
            return key
    return None


def _resolve_output_token_limit(chat_generator: ChatGenerator, default_limit: int) -> tuple[int, dict[str, Any] | None]:
    """
    Resolve an effective output-token limit and the runtime kwargs that hold a Chat Generator to it.

    A recognized limit already configured on the generator wins and is not repeated at runtime. This counts a limit the
    generator set for itself: `HuggingFaceAPIChatGenerator` and `TransformersChatGenerator` both leave a default of 512
    in their `generation_kwargs`, and reading the dict cannot tell that apart from a deliberate choice, so a caller
    asking for more than that gets the generator's number back.

    When a recognized generator has no limit configured, the default is returned as that provider's runtime setting. An
    unrecognized generator receives no runtime setting, because the `ChatGenerator` protocol guarantees nothing beyond
    `run(messages)` and guessing a key would raise a `TypeError` in the generator.

    :param chat_generator: The generator whose output should be limited.
    :param default_limit: The positive fallback output-token limit.
    :returns: The effective limit, and the `generation_kwargs` to pass at runtime, or None when there are none. A
        caller that gets None can still send the limit as prompt guidance and measure the reply itself.
    """
    limit_key = _generator_output_token_limit_key(chat_generator=chat_generator)
    if limit_key is None:
        return default_limit, None

    configured = getattr(chat_generator, "generation_kwargs", None)
    if isinstance(configured, dict) and limit_key in configured:
        value = configured[limit_key]
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value, None
        # The generator owns this setting. Do not silently replace an invalid value; let it report the problem.
        return default_limit, None
    return default_limit, {limit_key: default_limit}
