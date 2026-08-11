# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack.components.generators.chat.types import ChatGenerator

# The `generation_kwargs` key that caps a reply's length, per Chat Generator. Providers do not agree on a name and the
# `ChatGenerator` protocol does not standardize it. Keyed by class name because most of these live in
# `haystack-core-integrations` and cannot be imported here.
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
    # `__mro__` is the class followed by its bases, so an unlisted generator still matches through one it inherits
    # from, such as the integrations built on `OpenAIChatGenerator`. Most derived first, so its own entry wins.
    for cls in type(chat_generator).__mro__:
        key = _OUTPUT_TOKEN_LIMIT_KEYS.get(cls.__name__)
        if key is not None:
            return key
    return None


def _run_kwargs_with_output_limit(chat_generator: ChatGenerator, limit: int) -> dict[str, Any]:
    """
    Return the `run` kwargs that set a Chat Generator's output-token limit to `limit`.

    :param chat_generator: The generator whose output should be limited.
    :param limit: The positive output-token limit to set.
    :returns: The kwargs, or an empty dict when the generator is not recognized.
    """
    limit_key = _generator_output_token_limit_key(chat_generator=chat_generator)
    if limit_key is None:
        # Nothing at all rather than an empty `generation_kwargs`, which an unrecognized generator need not accept.
        return {}
    return {"generation_kwargs": {limit_key: limit}}
