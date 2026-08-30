# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.components.generators.utils import _normalize_messages, _trace_chat_generator_run
from haystack.core.serialization import component_to_dict
from haystack.dataclasses import ChatMessage, StreamingCallbackT
from haystack.document_stores.types import DocumentStore
from haystack.utils.async_utils import _execute_component_async
from haystack.utils.deserialization import deserialize_component_inplace

logger = logging.getLogger(__name__)

# ponytail: only params that change the LLM output.
_CACHE_RELEVANT_KEYS = frozenset(
    {"temperature", "top_p", "top_k", "max_tokens", "frequency_penalty", "presence_penalty", "seed", "stop", "n"}
)


def _compute_cache_key(messages: list[ChatMessage], generation_kwargs: dict[str, Any] | None = None) -> str:
    msg_dicts = [m.to_dict() for m in messages]
    filtered = {k: v for k, v in (generation_kwargs or {}).items() if k in _CACHE_RELEVANT_KEYS}
    data = {"messages": msg_dicts, "kwargs": filtered}
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()


@component
class LLMResponseCache:
    """
    Wraps a chat generator and caches responses in a DocumentStore.

    On cache hit, the underlying generator isn't called. Streaming responses
    bypass the cache since they can't be replayed.

    ```python
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.components.generators.chat.llm_response_cache import LLMResponseCache
    from haystack.document_stores.in_memory import InMemoryDocumentStore

    cache = LLMResponseCache(
        chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"),
        document_store=InMemoryDocumentStore(),
        ttl_seconds=3600,
    )
    result = cache.run(messages=[ChatMessage.from_user("What is 2+2?")])
    result = cache.run(messages=[ChatMessage.from_user("What is 2+2?")])  # cache hit
    assert result["meta"]["cache_hit"] is True
    ```
    """

    def __init__(self, chat_generator: Any, document_store: DocumentStore, ttl_seconds: int = 3600) -> None:
        self.chat_generator = chat_generator
        self.document_store = document_store
        self.ttl_seconds = ttl_seconds

    def to_dict(self) -> dict[str, Any]:
        """Serialize the component, including nested chat generator and document store."""
        return default_to_dict(
            self,
            chat_generator=component_to_dict(self.chat_generator, name="chat_generator"),
            document_store=component_to_dict(self.document_store, name="document_store"),
            ttl_seconds=self.ttl_seconds,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LLMResponseCache:
        """Rebuild the component from a serialized representation."""
        init_params = data.get("init_parameters", {})
        for key in ("chat_generator", "document_store"):
            serialized = init_params.get(key)
            if serialized is not None:
                holder = {"component": serialized}
                deserialize_component_inplace(holder, key="component")
                init_params[key] = holder["component"]
        data["init_parameters"] = init_params
        return default_from_dict(cls, data)

    def warm_up(self) -> None:
        """Warm up the underlying chat generator."""
        if hasattr(self.chat_generator, "warm_up"):
            self.chat_generator.warm_up()

    async def warm_up_async(self) -> None:
        """Warm up the underlying chat generator asynchronously."""
        if hasattr(self.chat_generator, "warm_up_async"):
            await self.chat_generator.warm_up_async()
        elif hasattr(self.chat_generator, "warm_up"):
            self.chat_generator.warm_up()

    def close(self) -> None:
        """Release the underlying resources."""
        if hasattr(self.chat_generator, "close"):
            self.chat_generator.close()
        if hasattr(self.document_store, "close"):
            self.document_store.close()

    async def close_async(self) -> None:
        """Release the underlying resources asynchronously."""
        if hasattr(self.chat_generator, "close_async"):
            await self.chat_generator.close_async()
        elif hasattr(self.chat_generator, "close"):
            self.chat_generator.close()
        if hasattr(self.document_store, "close_async"):
            await self.document_store.close_async()
        elif hasattr(self.document_store, "close"):
            self.document_store.close()

    def _lookup(self, cache_key: str) -> Document | None:
        """Find a non-expired cached entry for this key."""
        filters = {
            "operator": "AND",
            "conditions": [
                {"field": "meta.prompt_hash", "operator": "==", "value": cache_key},
                {"field": "meta.cached_at", "operator": ">=", "value": time.time() - self.ttl_seconds},
            ],
        }
        docs = self.document_store.filter_documents(filters=filters)
        return docs[0] if docs else None

    def _store(self, cache_key: str, replies: list[ChatMessage]) -> None:
        """Persist generator replies as cached Documents."""
        for reply in replies:
            doc = Document(
                content=reply.text or "",
                meta={
                    "prompt_hash": cache_key,
                    "cached_at": time.time(),
                    "model": reply.meta.get("model", ""),
                    "finish_reason": reply.meta.get("finish_reason", ""),
                    "usage": reply.meta.get("usage", {}),
                },
            )
            self.document_store.write_documents([doc])

    def _reconstruct(self, doc: Document) -> ChatMessage:
        """Build a ChatMessage from a cached Document."""
        return ChatMessage.from_assistant(
            text=doc.content or "",
            meta={
                "model": doc.meta.get("model", ""),
                "finish_reason": doc.meta.get("finish_reason", ""),
                "usage": doc.meta.get("usage", {}),
                "cache_hit": True,
            },
        )

    @component.output_types(replies=list[ChatMessage], meta=dict[str, Any])
    def run(
        self,
        messages: list[ChatMessage] | str,
        generation_kwargs: dict[str, Any] | None = None,
        streaming_callback: StreamingCallbackT | None = None,
    ) -> dict[str, list[ChatMessage] | dict[str, Any]]:
        """Check cache, forward to generator on miss. Streaming bypasses cache."""
        self.warm_up()
        messages = _normalize_messages(messages=messages)

        if streaming_callback is not None:
            inputs = {
                "messages": messages,
                "generation_kwargs": generation_kwargs,
                "streaming_callback": streaming_callback,
            }
            with _trace_chat_generator_run(chat_generator=self.chat_generator, generator_inputs=inputs):
                result = self.chat_generator.run(**inputs)
            return {"replies": result["replies"], "meta": {"cache_hit": False, "streaming": True}}

        cache_key = _compute_cache_key(messages, generation_kwargs)
        cached = self._lookup(cache_key)
        if cached is not None:
            return {"replies": [self._reconstruct(cached)], "meta": {"cache_hit": True, "cache_key": cache_key}}

        inputs = {"messages": messages, "generation_kwargs": generation_kwargs}
        with _trace_chat_generator_run(chat_generator=self.chat_generator, generator_inputs=inputs):
            result = self.chat_generator.run(**inputs)
        self._store(cache_key, result["replies"])
        return {"replies": result["replies"], "meta": {"cache_hit": False, "cache_key": cache_key}}

    @component.output_types(replies=list[ChatMessage], meta=dict[str, Any])
    async def run_async(
        self,
        messages: list[ChatMessage] | str,
        generation_kwargs: dict[str, Any] | None = None,
        streaming_callback: StreamingCallbackT | None = None,
    ) -> dict[str, list[ChatMessage] | dict[str, Any]]:
        """Async variant of run. Same behavior, same cache logic."""
        await self.warm_up_async()
        messages = _normalize_messages(messages=messages)

        if streaming_callback is not None:
            inputs = {
                "messages": messages,
                "generation_kwargs": generation_kwargs,
                "streaming_callback": streaming_callback,
            }
            with _trace_chat_generator_run(chat_generator=self.chat_generator, generator_inputs=inputs):
                result = await _execute_component_async(component_instance=self.chat_generator, **inputs)
            return {"replies": result["replies"], "meta": {"cache_hit": False, "streaming": True}}

        cache_key = _compute_cache_key(messages, generation_kwargs)
        cached = self._lookup(cache_key)
        if cached is not None:
            return {"replies": [self._reconstruct(cached)], "meta": {"cache_hit": True, "cache_key": cache_key}}

        inputs = {"messages": messages, "generation_kwargs": generation_kwargs}
        with _trace_chat_generator_run(chat_generator=self.chat_generator, generator_inputs=inputs):
            result = await _execute_component_async(component_instance=self.chat_generator, **inputs)
        self._store(cache_key, result["replies"])
        return {"replies": result["replies"], "meta": {"cache_hit": False, "cache_key": cache_key}}
