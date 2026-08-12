# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

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
    "AIMLAPIChatGenerator": "max_tokens",
    "AmazonBedrockChatGenerator": "maxTokens",
    "AnthropicChatGenerator": "max_tokens",
    "AnthropicFoundryChatGenerator": "max_tokens",
    "AnthropicVertexChatGenerator": "max_tokens",
    "CohereChatGenerator": "max_tokens",
    "CometAPIChatGenerator": "max_tokens",
    "EdenAIChatGenerator": "max_tokens",
    "GoogleAIGeminiChatGenerator": "max_output_tokens",
    "GoogleGenAIChatGenerator": "max_output_tokens",
    "HuggingFaceAPIChatGenerator": "max_tokens",
    "LiteLLMChatGenerator": "max_tokens",
    "LlamaCppChatGenerator": "max_tokens",
    "LlamaStackChatGenerator": "max_tokens",
    "MistralChatGenerator": "max_tokens",
    "NvidiaChatGenerator": "max_tokens",
    "OllamaChatGenerator": "num_predict",
    "OpenRouterChatGenerator": "max_tokens",
    "OrcaRouterChatGenerator": "max_tokens",
    "PerplexityChatGenerator": "max_output_tokens",
    "STACKITChatGenerator": "max_tokens",
    "TogetherAIChatGenerator": "max_tokens",
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
    return _OUTPUT_TOKEN_LIMIT_KEYS.get(type(chat_generator).__name__)
