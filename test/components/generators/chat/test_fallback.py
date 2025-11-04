# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import time
from typing import Any, Optional
from urllib.error import HTTPError as URLLibHTTPError

import pytest

from haystack import component, default_from_dict, default_to_dict
from haystack.components.generators.chat.fallback import FallbackChatGenerator
from haystack.dataclasses import ChatMessage, StreamingCallbackT
from haystack.tools import ToolsType


@component
class _DummySuccessGen:
    def __init__(self, text: str = "ok", delay: float = 0.0, streaming_callback: Optional[StreamingCallbackT] = None):
        self.text = text
        self.delay = delay
        self.streaming_callback = streaming_callback

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(self, text=self.text, delay=self.delay, streaming_callback=None)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "_DummySuccessGen":
        return default_from_dict(cls, data)

    def run(
        self,
        messages: list[ChatMessage],
        generation_kwargs: Optional[dict[str, Any]] = None,
        tools: Optional[ToolsType] = None,
        streaming_callback: Optional[StreamingCallbackT] = None,
    ) -> dict[str, Any]:
        if self.delay:
            time.sleep(self.delay)
        if streaming_callback:
            streaming_callback({"dummy": True})  # type: ignore[arg-type]
        return {"replies": [ChatMessage.from_assistant(self.text)], "meta": {"dummy_meta": True}}

    async def run_async(
        self,
        messages: list[ChatMessage],
        generation_kwargs: Optional[dict[str, Any]] = None,
        tools: Optional[ToolsType] = None,
        streaming_callback: Optional[StreamingCallbackT] = None,
    ) -> dict[str, Any]:
        if self.delay:
            await asyncio.sleep(self.delay)
        if streaming_callback:
            await asyncio.sleep(0)
            streaming_callback({"dummy": True})  # type: ignore[arg-type]
        return {"replies": [ChatMessage.from_assistant(self.text)], "meta": {"dummy_meta": True}}


@component
class _DummyFailGen:
    def __init__(self, exc: Optional[Exception] = None, delay: float = 0.0):
        self.exc = exc or RuntimeError("boom")
        self.delay = delay

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(self, exc={"message": str(self.exc)}, delay=self.delay)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "_DummyFailGen":
        init = data.get("init_parameters", {})
        msg = None
        if isinstance(init.get("exc"), dict):
            msg = init.get("exc", {}).get("message")
        return cls(exc=RuntimeError(msg or "boom"), delay=init.get("delay", 0.0))

    def run(
        self,
        messages: list[ChatMessage],
        generation_kwargs: Optional[dict[str, Any]] = None,
        tools: Optional[ToolsType] = None,
        streaming_callback: Optional[StreamingCallbackT] = None,
    ) -> dict[str, Any]:
        if self.delay:
            time.sleep(self.delay)
        raise self.exc

    async def run_async(
        self,
        messages: list[ChatMessage],
        generation_kwargs: Optional[dict[str, Any]] = None,
        tools: Optional[ToolsType] = None,
        streaming_callback: Optional[StreamingCallbackT] = None,
    ) -> dict[str, Any]:
        if self.delay:
            await asyncio.sleep(self.delay)
        raise self.exc


def test_init_validation():
    with pytest.raises(ValueError):
        FallbackChatGenerator(chat_generators=[])

    gen = FallbackChatGenerator(chat_generators=[_DummySuccessGen(text="A")])
    assert len(gen.chat_generators) == 1


def test_sequential_first_success():
    gen = FallbackChatGenerator(chat_generators=[_DummySuccessGen(text="A")])
    res = gen.run([ChatMessage.from_user("hi")])
    assert res["replies"][0].text == "A"
    assert res["meta"]["successful_chat_generator_index"] == 0
    assert res["meta"]["total_attempts"] == 1


def test_sequential_second_success_after_failure():
    gen = FallbackChatGenerator(chat_generators=[_DummyFailGen(), _DummySuccessGen(text="B")])
    res = gen.run([ChatMessage.from_user("hi")])
    assert res["replies"][0].text == "B"
    assert res["meta"]["successful_chat_generator_index"] == 1
    assert res["meta"]["failed_chat_generators"]


def test_all_fail_raises():
    gen = FallbackChatGenerator(chat_generators=[_DummyFailGen(), _DummyFailGen()])
    with pytest.raises(RuntimeError):
        gen.run([ChatMessage.from_user("hi")])


def test_timeout_handling_sync():
    slow = _DummySuccessGen(text="slow", delay=0.01)
    fast = _DummySuccessGen(text="fast", delay=0.0)
    gen = FallbackChatGenerator(chat_generators=[slow, fast])
    res = gen.run([ChatMessage.from_user("hi")])
    assert res["replies"][0].text == "slow"


@pytest.mark.asyncio
async def test_timeout_handling_async():
    slow = _DummySuccessGen(text="slow", delay=0.01)
    fast = _DummySuccessGen(text="fast", delay=0.0)
    gen = FallbackChatGenerator(chat_generators=[slow, fast])
    res = await gen.run_async([ChatMessage.from_user("hi")])
    assert res["replies"][0].text == "slow"


def test_streaming_callback_forwarding_sync():
    calls: list[Any] = []

    def cb(x: Any) -> None:
        calls.append(x)

    gen = FallbackChatGenerator(chat_generators=[_DummySuccessGen(text="A")])
    _ = gen.run([ChatMessage.from_user("hi")], streaming_callback=cb)
    assert calls


@pytest.mark.asyncio
async def test_streaming_callback_forwarding_async():
    calls: list[Any] = []

    def cb(x: Any) -> None:
        calls.append(x)

    gen = FallbackChatGenerator(chat_generators=[_DummySuccessGen(text="A")])
    _ = await gen.run_async([ChatMessage.from_user("hi")], streaming_callback=cb)
    assert calls


def test_serialization_roundtrip():
    original = FallbackChatGenerator(chat_generators=[_DummySuccessGen(text="hello")])
    data = original.to_dict()
    restored = FallbackChatGenerator.from_dict(data)
    assert isinstance(restored, FallbackChatGenerator)
    assert len(restored.chat_generators) == 1
    res = restored.run([ChatMessage.from_user("hi")])
    assert res["replies"][0].text == "hello"

    original = FallbackChatGenerator(chat_generators=[_DummySuccessGen(text="hello"), _DummySuccessGen(text="world")])
    data = original.to_dict()
    restored = FallbackChatGenerator.from_dict(data)
    assert isinstance(restored, FallbackChatGenerator)
    assert len(restored.chat_generators) == 2
    res = restored.run([ChatMessage.from_user("hi")])
    assert res["replies"][0].text == "hello"


def test_automatic_completion_mode_without_streaming():
    gen = FallbackChatGenerator(chat_generators=[_DummySuccessGen(text="completion")])
    res = gen.run([ChatMessage.from_user("hi")])
    assert res["replies"][0].text == "completion"
    assert res["meta"]["successful_chat_generator_index"] == 0


def test_automatic_ttft_mode_with_streaming():
    calls: list[Any] = []

    def cb(x: Any) -> None:
        calls.append(x)

    gen = FallbackChatGenerator(chat_generators=[_DummySuccessGen(text="streaming")])
    res = gen.run([ChatMessage.from_user("hi")], streaming_callback=cb)
    assert res["replies"][0].text == "streaming"
    assert calls


@pytest.mark.asyncio
async def test_automatic_ttft_mode_with_streaming_async():
    calls: list[Any] = []

    def cb(x: Any) -> None:
        calls.append(x)

    gen = FallbackChatGenerator(chat_generators=[_DummySuccessGen(text="streaming_async")])
    res = await gen.run_async([ChatMessage.from_user("hi")], streaming_callback=cb)
    assert res["replies"][0].text == "streaming_async"
    assert calls


def create_http_error(status_code: int, message: str) -> URLLibHTTPError:
    return URLLibHTTPError("", status_code, message, {}, None)


@component
class _DummyHTTPErrorGen:
    def __init__(self, text: str = "success", error: Optional[Exception] = None):
        self.text = text
        self.error = error

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(self, text=self.text, error=str(self.error) if self.error else None)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "_DummyHTTPErrorGen":
        init = data.get("init_parameters", {})
        error = None
        if init.get("error"):
            error = RuntimeError(init["error"])
        return cls(text=init.get("text", "success"), error=error)

    def run(
        self,
        messages: list[ChatMessage],
        generation_kwargs: Optional[dict[str, Any]] = None,
        tools: Optional[ToolsType] = None,
        streaming_callback: Optional[StreamingCallbackT] = None,
    ) -> dict[str, Any]:
        if self.error:
            raise self.error
        return {
            "replies": [ChatMessage.from_assistant(self.text)],
            "meta": {"error_type": type(self.error).__name__ if self.error else None},
        }


def test_failover_trigger_429_rate_limit():
    rate_limit_gen = _DummyHTTPErrorGen(text="rate_limited", error=create_http_error(429, "Rate limit exceeded"))
    success_gen = _DummySuccessGen(text="success_after_rate_limit")

    fallback = FallbackChatGenerator(chat_generators=[rate_limit_gen, success_gen])
    result = fallback.run([ChatMessage.from_user("test")])

    assert result["replies"][0].text == "success_after_rate_limit"
    assert result["meta"]["successful_chat_generator_index"] == 1
    assert result["meta"]["failed_chat_generators"] == ["_DummyHTTPErrorGen"]


def test_failover_trigger_401_authentication():
    auth_error_gen = _DummyHTTPErrorGen(text="auth_failed", error=create_http_error(401, "Authentication failed"))
    success_gen = _DummySuccessGen(text="success_after_auth")

    fallback = FallbackChatGenerator(chat_generators=[auth_error_gen, success_gen])
    result = fallback.run([ChatMessage.from_user("test")])

    assert result["replies"][0].text == "success_after_auth"
    assert result["meta"]["successful_chat_generator_index"] == 1
    assert result["meta"]["failed_chat_generators"] == ["_DummyHTTPErrorGen"]


def test_failover_trigger_400_bad_request():
    bad_request_gen = _DummyHTTPErrorGen(text="bad_request", error=create_http_error(400, "Context length exceeded"))
    success_gen = _DummySuccessGen(text="success_after_bad_request")

    fallback = FallbackChatGenerator(chat_generators=[bad_request_gen, success_gen])
    result = fallback.run([ChatMessage.from_user("test")])

    assert result["replies"][0].text == "success_after_bad_request"
    assert result["meta"]["successful_chat_generator_index"] == 1
    assert result["meta"]["failed_chat_generators"] == ["_DummyHTTPErrorGen"]


def test_failover_trigger_500_server_error():
    server_error_gen = _DummyHTTPErrorGen(text="server_error", error=create_http_error(500, "Internal server error"))
    success_gen = _DummySuccessGen(text="success_after_server_error")

    fallback = FallbackChatGenerator(chat_generators=[server_error_gen, success_gen])
    result = fallback.run([ChatMessage.from_user("test")])

    assert result["replies"][0].text == "success_after_server_error"
    assert result["meta"]["successful_chat_generator_index"] == 1
    assert result["meta"]["failed_chat_generators"] == ["_DummyHTTPErrorGen"]


def test_failover_trigger_multiple_errors():
    rate_limit_gen = _DummyHTTPErrorGen(text="rate_limited", error=create_http_error(429, "Rate limit exceeded"))
    auth_error_gen = _DummyHTTPErrorGen(text="auth_failed", error=create_http_error(401, "Authentication failed"))
    server_error_gen = _DummyHTTPErrorGen(text="server_error", error=create_http_error(500, "Internal server error"))
    success_gen = _DummySuccessGen(text="success_after_all_errors")

    fallback = FallbackChatGenerator(chat_generators=[rate_limit_gen, auth_error_gen, server_error_gen, success_gen])
    result = fallback.run([ChatMessage.from_user("test")])

    assert result["replies"][0].text == "success_after_all_errors"
    assert result["meta"]["successful_chat_generator_index"] == 3
    assert len(result["meta"]["failed_chat_generators"]) == 3


def test_failover_trigger_all_generators_fail():
    rate_limit_gen = _DummyHTTPErrorGen(text="rate_limited", error=create_http_error(429, "Rate limit exceeded"))
    auth_error_gen = _DummyHTTPErrorGen(text="auth_failed", error=create_http_error(401, "Authentication failed"))
    server_error_gen = _DummyHTTPErrorGen(text="server_error", error=create_http_error(500, "Internal server error"))

    fallback = FallbackChatGenerator(chat_generators=[rate_limit_gen, auth_error_gen, server_error_gen])

    with pytest.raises(RuntimeError) as exc_info:
        fallback.run([ChatMessage.from_user("test")])

    error_msg = str(exc_info.value)
    assert "All 3 chat generators failed" in error_msg
    assert "Failed chat generators: [_DummyHTTPErrorGen, _DummyHTTPErrorGen, _DummyHTTPErrorGen]" in error_msg


@pytest.mark.asyncio
async def test_failover_trigger_429_rate_limit_async():
    rate_limit_gen = _DummyHTTPErrorGen(text="rate_limited", error=create_http_error(429, "Rate limit exceeded"))
    success_gen = _DummySuccessGen(text="success_after_rate_limit")

    fallback = FallbackChatGenerator(chat_generators=[rate_limit_gen, success_gen])
    result = await fallback.run_async([ChatMessage.from_user("test")])

    assert result["replies"][0].text == "success_after_rate_limit"
    assert result["meta"]["successful_chat_generator_index"] == 1
    assert result["meta"]["failed_chat_generators"] == ["_DummyHTTPErrorGen"]


@pytest.mark.asyncio
async def test_failover_trigger_401_authentication_async():
    auth_error_gen = _DummyHTTPErrorGen(text="auth_failed", error=create_http_error(401, "Authentication failed"))
    success_gen = _DummySuccessGen(text="success_after_auth")

    fallback = FallbackChatGenerator(chat_generators=[auth_error_gen, success_gen])
    result = await fallback.run_async([ChatMessage.from_user("test")])

    assert result["replies"][0].text == "success_after_auth"
    assert result["meta"]["successful_chat_generator_index"] == 1
    assert result["meta"]["failed_chat_generators"] == ["_DummyHTTPErrorGen"]


@component
class _DummyGenWithWarmUp:
    """Dummy generator that tracks warm_up calls."""

    def __init__(self, text: str = "ok"):
        self.text = text
        self.warm_up_called = False

    def warm_up(self) -> None:
        self.warm_up_called = True

    def run(
        self,
        messages: list[ChatMessage],
        generation_kwargs: Optional[dict[str, Any]] = None,
        tools: Optional[ToolsType] = None,
        streaming_callback: Optional[StreamingCallbackT] = None,
    ) -> dict[str, Any]:
        return {"replies": [ChatMessage.from_assistant(self.text)], "meta": {}}


def test_warm_up_delegates_to_generators():
    """Test that warm_up() is called on each underlying generator."""
    gen1 = _DummyGenWithWarmUp(text="A")
    gen2 = _DummyGenWithWarmUp(text="B")
    gen3 = _DummyGenWithWarmUp(text="C")

    fallback = FallbackChatGenerator(chat_generators=[gen1, gen2, gen3])
    fallback.warm_up()

    assert gen1.warm_up_called
    assert gen2.warm_up_called
    assert gen3.warm_up_called


def test_warm_up_with_no_warm_up_method():
    """Test that warm_up() handles generators without warm_up() gracefully."""
    gen1 = _DummySuccessGen(text="A")
    gen2 = _DummySuccessGen(text="B")

    fallback = FallbackChatGenerator(chat_generators=[gen1, gen2])
    # Should not raise any error
    fallback.warm_up()

    # Verify generators still work
    result = fallback.run([ChatMessage.from_user("test")])
    assert result["replies"][0].text == "A"


def test_warm_up_mixed_generators():
    """Test warm_up() with a mix of generators with and without warm_up()."""
    gen1 = _DummyGenWithWarmUp(text="A")
    gen2 = _DummySuccessGen(text="B")
    gen3 = _DummyGenWithWarmUp(text="C")
    gen4 = _DummyFailGen()

    fallback = FallbackChatGenerator(chat_generators=[gen1, gen2, gen3, gen4])
    fallback.warm_up()

    # Only generators with warm_up() should have been called
    assert gen1.warm_up_called
    assert gen3.warm_up_called

    # Verify the fallback still works correctly
    result = fallback.run([ChatMessage.from_user("test")])
    assert result["replies"][0].text == "A"
