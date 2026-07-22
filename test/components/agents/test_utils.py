# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack.components.agents.state import State, replace_values
from haystack.components.agents.utils import _context_tokens_from_usage, _record_context_tokens
from haystack.dataclasses import ChatMessage


class TestContextTokensFromUsage:
    """`_context_tokens_from_usage` normalizes real provider `meta["usage"]` shapes to input + output tokens."""

    # OpenAI Chat Completions (core repo): reasoning_tokens is a subset of completion_tokens (64 of 74), so
    # context_tokens includes reasoning.
    OPENAI_CHAT_USAGE = {
        "completion_tokens": 74,
        "prompt_tokens": 19,
        "total_tokens": 93,
        "completion_tokens_details": {
            "accepted_prediction_tokens": 0,
            "audio_tokens": 0,
            "reasoning_tokens": 64,
            "rejected_prediction_tokens": 0,
        },
        "prompt_tokens_details": {"audio_tokens": 0, "cached_tokens": 0},
    }
    # OpenAI Responses API (core repo).
    OPENAI_RESPONSES_USAGE = {
        "input_tokens": 19,
        "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
        "output_tokens": 58,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 77,
    }
    # Anthropic chat generator (integration): no total_tokens, and thinking_tokens is a subset of completion_tokens
    # (57 of 63).
    ANTHROPIC_USAGE = {
        "cache_creation": {"ephemeral_1h_input_tokens": 0, "ephemeral_5m_input_tokens": 0},
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "inference_geo": "not_available",
        "output_tokens_details": {"thinking_tokens": 57},
        "server_tool_use": None,
        "service_tier": "standard",
        "prompt_tokens": 49,
        "completion_tokens": 63,
    }
    # Amazon Bedrock chat generator (integration).
    BEDROCK_USAGE = {
        "prompt_tokens": 20,
        "completion_tokens": 5,
        "total_tokens": 25,
        "cache_read_input_tokens": 0,
        "cache_write_input_tokens": 0,
        "cache_details": {},
    }
    # Cohere chat generator (integration).
    COHERE_USAGE = {"prompt_tokens": 15.0, "completion_tokens": 3.0}
    # Mistral chat generator (integration).
    MISTRAL_USAGE = {
        "prompt_tokens": 30,
        "total_tokens": 34,
        "completion_tokens": 4,
        "prompt_tokens_details": {"cached_tokens": 0},
    }
    # Nvidia chat generator (integration).
    NVIDIA_USAGE = {
        "completion_tokens": 2,
        "prompt_tokens": 48,
        "total_tokens": 50,
        "completion_tokens_details": None,
        "prompt_tokens_details": {"audio_tokens": None, "cached_tokens": 16},
    }
    # Google GenAI chat generator (integration): thoughts_token_count (281) is NOT part of completion_tokens;
    # total_tokens (300) includes it, so context_tokens (19) excludes thoughts.
    GOOGLE_GENAI_USAGE = {
        "prompt_tokens": 16,
        "completion_tokens": 3,
        "total_tokens": 300,
        "thoughts_token_count": 281,
        "prompt_token_count": 16,
        "candidates_token_count": 3,
        "total_token_count": 300,
        "prompt_tokens_details": [{"modality": "TEXT", "token_count": 16}],
    }

    @pytest.mark.parametrize(
        "usage, expected",
        [
            (OPENAI_CHAT_USAGE, 93),
            (OPENAI_RESPONSES_USAGE, 77),
            (ANTHROPIC_USAGE, 112),
            (BEDROCK_USAGE, 25),
            (COHERE_USAGE, 18),
            (MISTRAL_USAGE, 34),
            (NVIDIA_USAGE, 50),
            (GOOGLE_GENAI_USAGE, 19),  # 16 + 3, deliberately not the 300 total (which includes 281 thoughts tokens)
            ({"prompt_tokens": 10}, 10),  # only one side reported -> partial count
            ({"completion_tokens": 7}, 7),
            ({}, 0),  # no usage reported
            ({"foo": 5, "bar": 9}, 0),  # no recognized keys
        ],
    )
    def test_normalizes_provider_shapes(self, usage, expected):
        assert _context_tokens_from_usage(usage) == expected

    def test_bool_values_are_not_counted_as_tokens(self):
        # bool is an int subclass; True/False under a token key must be skipped, not summed.
        assert _context_tokens_from_usage({"prompt_tokens": True, "completion_tokens": 5}) == 5


class TestRecordContextTokens:
    """`_record_context_tokens` replaces the value with the latest reply's input+output, only when usage is reported."""

    def _state(self) -> State:
        state = State(schema={"context_tokens": {"type": int, "handler": replace_values}})
        state.set("context_tokens", 0)
        return state

    def test_records_latest_reply_usage_replacing_previous_value(self):
        state = self._state()
        state.set("context_tokens", 999)
        _record_context_tokens(
            state, [ChatMessage.from_assistant("Hi", meta={"usage": {"prompt_tokens": 12, "completion_tokens": 3}})]
        )
        assert state.get("context_tokens") == 15

    def test_no_messages_leaves_value_untouched(self):
        state = self._state()
        state.set("context_tokens", 42)
        _record_context_tokens(state, [])
        assert state.get("context_tokens") == 42

    def test_missing_or_empty_usage_leaves_value_untouched(self):
        state = self._state()
        _record_context_tokens(state, [ChatMessage.from_assistant("no usage here")])
        _record_context_tokens(state, [ChatMessage.from_assistant("empty", meta={"usage": {}})])
        assert state.get("context_tokens") == 0
