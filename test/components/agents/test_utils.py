# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack.components.agents.state import State, replace_values
from haystack.components.agents.utils import _context_tokens_from_usage, _record_context_tokens
from haystack.dataclasses import ChatMessage


class TestContextTokensFromUsage:
    """`_context_tokens_from_usage` normalizes real provider `meta["usage"]` shapes to input + output tokens.

    Fixtures below are the actual shapes these providers report (nested detail dicts and all), so the test pins
    the normalization against reality rather than a simplified stand-in.
    """

    # OpenAI Chat Completions (core repo): `_serialize_object(completion.usage)` -> the full CompletionUsage dump.
    OPENAI_CHAT_USAGE = {
        "prompt_tokens": 1117,
        "completion_tokens": 46,
        "total_tokens": 1163,
        "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
        "completion_tokens_details": {
            "reasoning_tokens": 0,
            "audio_tokens": 0,
            "accepted_prediction_tokens": 0,
            "rejected_prediction_tokens": 0,
        },
    }
    # OpenAI Responses API (core repo): the only Haystack generator that uses input/output keys, each with a
    # details dict.
    OPENAI_RESPONSES_USAGE = {
        "input_tokens": 75,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 1186,
        "output_tokens_details": {"reasoning_tokens": 1024},
        "total_tokens": 1261,
    }
    # Anthropic chat generator (integration): remaps input/output -> prompt/completion and keeps the cache
    # accounting (incl. the nested cache_creation breakdown).
    ANTHROPIC_USAGE = {
        "prompt_tokens": 2095,
        "completion_tokens": 503,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 1024,
        "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
        "service_tier": "standard",
    }
    # Amazon Bedrock chat generator (integration): prompt/completion/total plus cache accounting.
    BEDROCK_USAGE = {
        "prompt_tokens": 340,
        "completion_tokens": 92,
        "total_tokens": 432,
        "cache_read_input_tokens": 0,
        "cache_write_input_tokens": 0,
        "cache_details": {},
    }
    # Google GenAI chat generator (integration): total_tokens folds in thoughts, so it exceeds prompt + completion.
    # We deliberately use prompt+completion, so thoughts are excluded.
    GOOGLE_GENAI_USAGE = {
        "prompt_tokens": 20,
        "completion_tokens": 50,
        "total_tokens": 85,
        "thoughts_token_count": 15,
        "cached_content_token_count": 0,
    }

    @pytest.mark.parametrize(
        "usage, expected",
        [
            (OPENAI_CHAT_USAGE, 1163),
            (OPENAI_RESPONSES_USAGE, 1261),
            (ANTHROPIC_USAGE, 2598),
            (BEDROCK_USAGE, 432),
            (GOOGLE_GENAI_USAGE, 70),  # 20 + 50, deliberately not the 85 total (which includes 15 thoughts tokens)
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

    def test_records_latest_reply_usage(self):
        state = self._state()
        _record_context_tokens(
            state, [ChatMessage.from_assistant("Hi", meta={"usage": {"prompt_tokens": 12, "completion_tokens": 3}})]
        )
        assert state.get("context_tokens") == 15

    def test_replaces_rather_than_accumulates(self):
        state = self._state()
        state.set("context_tokens", 999)
        _record_context_tokens(
            state, [ChatMessage.from_assistant("Hi", meta={"usage": {"prompt_tokens": 4, "completion_tokens": 1}})]
        )
        assert state.get("context_tokens") == 5

    def test_no_messages_leaves_value_untouched(self):
        state = self._state()
        state.set("context_tokens", 42)
        _record_context_tokens(state, [])
        assert state.get("context_tokens") == 42

    def test_missing_or_empty_usage_leaves_value_untouched(self):
        # A reply without usage (or with empty usage) must not clobber the previous value, including the initial 0.
        state = self._state()
        _record_context_tokens(state, [ChatMessage.from_assistant("no usage here")])
        _record_context_tokens(state, [ChatMessage.from_assistant("empty", meta={"usage": {}})])
        assert state.get("context_tokens") == 0
