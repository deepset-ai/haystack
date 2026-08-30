# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock

import pytest

from haystack.components.generators.chat.llm_response_cache import LLMResponseCache, _compute_cache_key
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore


@pytest.fixture
def document_store():
    return InMemoryDocumentStore()


@pytest.fixture
def mock_chat_generator():
    gen = MagicMock()
    gen.run.return_value = {
        "replies": [
            ChatMessage.from_assistant(
                text="4",
                meta={
                    "model": "gpt-4o",
                    "finish_reason": "stop",
                    "usage": {"prompt_tokens": 5, "completion_tokens": 1},
                },
            )
        ]
    }
    gen.warm_up = MagicMock()
    gen.warm_up_async = AsyncMock()
    del gen.run_async  # so _execute_component_async uses sync run in a thread
    return gen


@pytest.fixture
def cache(document_store, mock_chat_generator):
    return LLMResponseCache(chat_generator=mock_chat_generator, document_store=document_store, ttl_seconds=3600)


class TestComputeCacheKey:
    def test_deterministic(self):
        msgs = [ChatMessage.from_user("hello")]
        assert _compute_cache_key(msgs) == _compute_cache_key(msgs)

    def test_different_messages_different_keys(self):
        assert _compute_cache_key([ChatMessage.from_user("hello")]) != _compute_cache_key(
            [ChatMessage.from_user("world")]
        )

    def test_generation_kwargs_affect_key(self):
        msgs = [ChatMessage.from_user("hello")]
        assert _compute_cache_key(msgs, {"temperature": 0.0}) != _compute_cache_key(msgs, {"temperature": 1.0})

    def test_irrelevant_kwargs_ignored(self):
        msgs = [ChatMessage.from_user("hello")]
        assert _compute_cache_key(msgs, {"api_key": "secret"}) == _compute_cache_key(msgs)


class TestCacheHit:
    def test_first_call_misses(self, cache, mock_chat_generator):
        result = cache.run(messages=[ChatMessage.from_user("What is 2+2?")])
        assert result["meta"]["cache_hit"] is False
        mock_chat_generator.run.assert_called_once()

    def test_second_call_hits(self, cache, mock_chat_generator):
        cache.run(messages=[ChatMessage.from_user("What is 2+2?")])
        mock_chat_generator.run.reset_mock()
        result = cache.run(messages=[ChatMessage.from_user("What is 2+2?")])
        assert result["meta"]["cache_hit"] is True
        mock_chat_generator.run.assert_not_called()

    def test_cached_response_text(self, cache):
        cache.run(messages=[ChatMessage.from_user("What is 2+2?")])
        result = cache.run(messages=[ChatMessage.from_user("What is 2+2?")])
        assert result["replies"][0].text == "4"

    def test_different_prompts_different_entries(self, cache, mock_chat_generator):
        cache.run(messages=[ChatMessage.from_user("What is 2+2?")])
        cache.run(messages=[ChatMessage.from_user("What is 3+3?")])
        assert mock_chat_generator.run.call_count == 2


class TestTTLExpiry:
    def test_expired_entry_not_returned(self, document_store, mock_chat_generator):
        cache = LLMResponseCache(chat_generator=mock_chat_generator, document_store=document_store, ttl_seconds=0)
        cache.run(messages=[ChatMessage.from_user("hello")])
        mock_chat_generator.run.reset_mock()
        result = cache.run(messages=[ChatMessage.from_user("hello")])
        assert result["meta"]["cache_hit"] is False  # type: ignore[call-overload]
        mock_chat_generator.run.assert_called_once()

    def test_fresh_entry_returned(self, cache):
        cache.run(messages=[ChatMessage.from_user("hello")])
        result = cache.run(messages=[ChatMessage.from_user("hello")])
        assert result["meta"]["cache_hit"] is True


class TestStreamingBypass:
    def test_streaming_skips_cache(self, cache, mock_chat_generator):
        cb = MagicMock()
        cache.run(messages=[ChatMessage.from_user("hello")], streaming_callback=cb)
        mock_chat_generator.run.reset_mock()
        result = cache.run(messages=[ChatMessage.from_user("hello")], streaming_callback=cb)
        assert result["meta"]["cache_hit"] is False
        assert result["meta"]["streaming"] is True
        mock_chat_generator.run.assert_called_once()


class TestAsync:
    @pytest.mark.asyncio
    async def test_async_cache_miss(self, cache, mock_chat_generator):
        mock_chat_generator.run = MagicMock(return_value={"replies": [ChatMessage.from_assistant(text="async")]})
        result = await cache.run_async(messages=[ChatMessage.from_user("hello")])
        assert result["meta"]["cache_hit"] is False
        mock_chat_generator.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_cache_hit(self, cache, mock_chat_generator):
        cache.run(messages=[ChatMessage.from_user("hello")])
        mock_chat_generator.run.reset_mock()
        result = await cache.run_async(messages=[ChatMessage.from_user("hello")])
        assert result["meta"]["cache_hit"] is True
        mock_chat_generator.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_async_streaming_skips_cache(self, cache, mock_chat_generator):
        cb = MagicMock()
        await cache.run_async(messages=[ChatMessage.from_user("hello")], streaming_callback=cb)
        mock_chat_generator.run.reset_mock()
        result = await cache.run_async(messages=[ChatMessage.from_user("hello")], streaming_callback=cb)
        assert result["meta"]["cache_hit"] is False
        assert result["meta"]["streaming"] is True


class TestSerialization:
    def test_to_dict_round_trip(self, document_store, mock_chat_generator):
        cache = LLMResponseCache(chat_generator=mock_chat_generator, document_store=document_store, ttl_seconds=7200)
        serialized = cache.to_dict()
        assert serialized["init_parameters"]["ttl_seconds"] == 7200
        assert "chat_generator" in serialized["init_parameters"]
        assert "document_store" in serialized["init_parameters"]

    def test_from_dict_round_trip(self):
        from haystack.components.generators.chat.mock import MockChatGenerator

        cache = LLMResponseCache(
            chat_generator=MockChatGenerator(), document_store=InMemoryDocumentStore(), ttl_seconds=1800
        )
        restored = LLMResponseCache.from_dict(cache.to_dict())
        assert restored.ttl_seconds == 1800


class TestLifecycle:
    def test_warm_up(self, cache, mock_chat_generator):
        cache.warm_up()
        mock_chat_generator.warm_up.assert_called_once()

    def test_close(self, cache, mock_chat_generator, document_store):
        mock_chat_generator.close = MagicMock()
        document_store.close = MagicMock()
        cache.close()
        mock_chat_generator.close.assert_called_once()
        document_store.close.assert_called_once()

    def test_warm_up_safe_without_method(self, document_store):
        cache = LLMResponseCache(chat_generator=MagicMock(spec=[]), document_store=document_store)
        cache.warm_up()
