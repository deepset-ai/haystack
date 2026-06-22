# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import inspect

import pytest

from haystack import Pipeline
from haystack.components.generators.chat import MockChatGenerator
from haystack.dataclasses import ChatMessage, StreamingChunk, ToolCall


def _exclaim(messages: list[ChatMessage]) -> str:
    """Module-level response function (returns a string) used to test `response_fn` and its serialization."""
    return f"{messages[-1].text}!"


def _assistant_reply(messages: list[ChatMessage]) -> ChatMessage:
    """Module-level response function that returns a full ChatMessage."""
    return ChatMessage.from_assistant("canned message")


def _noop_callback(chunk: StreamingChunk) -> None:
    """Module-level streaming callback used to test init-level callback serialization."""


class TestMockChatGenerator:
    @pytest.mark.parametrize(
        ("args", "kwargs", "exception", "match"),
        [
            (("a",), {"response_fn": _exclaim}, ValueError, "either 'responses' or 'response_fn'"),
            (([],), {}, ValueError, "must not be an empty list"),
            ((123,), {}, TypeError, "must be a string, ChatMessage, or a sequence"),
            (([123],), {}, TypeError, "Each response must be a string or ChatMessage"),
        ],
    )
    def test_init_rejects_invalid_config(self, args, kwargs, exception, match):
        with pytest.raises(exception, match=match):
            MockChatGenerator(*args, **kwargs)

    def test_fixed_response(self):
        gen = MockChatGenerator("the same answer")
        for _ in range(3):
            result = gen.run([ChatMessage.from_user("anything")])
            assert result["replies"][0].text == "the same answer"

    def test_cycling_responses(self):
        # a mix of strings and ChatMessage objects, returned in order and wrapping around
        gen = MockChatGenerator(["one", ChatMessage.from_assistant("two"), "three"])
        texts = [gen.run([ChatMessage.from_user("hi")])["replies"][0].text for _ in range(4)]
        assert texts == ["one", "two", "three", "one"]

    @pytest.mark.parametrize(
        ("messages", "expected"),
        [
            (
                [ChatMessage.from_system("sys"), ChatMessage.from_user("first"), ChatMessage.from_user("second")],
                "second",
            ),
            ([ChatMessage.from_system("only system")], "only system"),  # falls back to the last message with text
            ([], None),  # nothing to echo
        ],
    )
    def test_echo_default(self, messages, expected):
        replies = MockChatGenerator().run(messages)["replies"]
        if expected is None:
            assert replies == []
        else:
            assert replies[0].text == expected

    @pytest.mark.parametrize(("fn", "expected"), [(_exclaim, "hello!"), (_assistant_reply, "canned message")])
    def test_response_fn(self, fn, expected):
        result = MockChatGenerator(response_fn=fn).run([ChatMessage.from_user("hello")])
        assert result["replies"][0].text == expected

    def test_response_fn_invalid_return_raises(self):
        gen = MockChatGenerator(response_fn=lambda messages: 123)
        with pytest.raises(TypeError, match="must return a string or ChatMessage"):
            gen.run([ChatMessage.from_user("hi")])

    def test_string_input_is_normalized(self):
        gen = MockChatGenerator(response_fn=_exclaim)
        assert gen.run("plain string")["replies"][0].text == "plain string!"

    def test_tool_call_response(self):
        tool_call = ToolCall(tool_name="search", arguments={"query": "Haystack"})
        gen = MockChatGenerator(ChatMessage.from_assistant(tool_calls=[tool_call]))
        reply = gen.run([ChatMessage.from_user("search for Haystack")])["replies"][0]
        assert reply.tool_calls == [tool_call]
        assert reply.meta["finish_reason"] == "tool_calls"

    def test_meta_defaults(self):
        meta = MockChatGenerator("hello world").run([ChatMessage.from_user("a b c")])["replies"][0].meta
        assert meta["model"] == "mock-model"
        assert meta["finish_reason"] == "stop"
        assert meta["usage"] == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}

    def test_meta_merging_precedence(self):
        # init meta overrides defaults; per-response meta overrides init meta
        response = ChatMessage.from_assistant("hi", meta={"custom": "from-response", "finish_reason": "length"})
        gen = MockChatGenerator(response, model="custom-model", meta={"custom": "from-init", "extra": "init"})
        meta = gen.run([ChatMessage.from_user("x")])["replies"][0].meta
        assert meta["model"] == "custom-model"
        assert meta["custom"] == "from-response"
        assert meta["finish_reason"] == "length"
        assert meta["extra"] == "init"

    def test_does_not_mutate_stored_responses(self):
        gen = MockChatGenerator("hello")
        gen.run([ChatMessage.from_user("a b")])
        # the stored response keeps its original (empty) meta, untouched by the per-run meta
        assert gen._responses[0].meta == {}

    async def test_run_async(self):
        gen = MockChatGenerator(["one", "two"])
        assert (await gen.run_async([ChatMessage.from_user("hi")]))["replies"][0].text == "one"
        assert (await gen.run_async([ChatMessage.from_user("hi")]))["replies"][0].text == "two"
        # echo mode with empty input returns no replies (async path)
        assert (await MockChatGenerator().run_async([]))["replies"] == []

    def test_streaming_callback_sync(self):
        chunks: list[StreamingChunk] = []
        result = MockChatGenerator("hello there friend").run(
            [ChatMessage.from_user("hi")], streaming_callback=chunks.append
        )
        assert "".join(chunk.content for chunk in chunks) == "hello there friend"
        assert chunks[0].start is True
        assert chunks[-1].finish_reason == "stop"
        # the returned reply matches the predefined response
        assert result["replies"][0].text == "hello there friend"

    def test_run_signature_matches_openai_order(self):
        # run()/run_async() must mirror OpenAIChatGenerator's parameter order so the mock is a positional drop-in.
        expected = [
            ("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            ("messages", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            ("streaming_callback", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            ("generation_kwargs", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            ("tools", inspect.Parameter.KEYWORD_ONLY),
            ("tools_strict", inspect.Parameter.KEYWORD_ONLY),
        ]
        for method in ("run", "run_async"):
            params = list(inspect.signature(getattr(MockChatGenerator, method)).parameters.values())
            assert [(p.name, p.kind) for p in params] == expected

        # passing the callback as the 2nd positional arg must be treated as streaming_callback, not generation_kwargs
        chunks: list[StreamingChunk] = []
        MockChatGenerator("hi").run([ChatMessage.from_user("x")], chunks.append)
        assert chunks

    async def test_streaming_callback_async(self):
        chunks: list[StreamingChunk] = []

        async def callback(chunk: StreamingChunk) -> None:
            chunks.append(chunk)

        await MockChatGenerator("hello world").run_async([ChatMessage.from_user("hi")], streaming_callback=callback)
        assert "".join(chunk.content for chunk in chunks) == "hello world"
        assert chunks[-1].finish_reason == "stop"

    def test_streaming_empty_reply(self):
        chunks: list[StreamingChunk] = []
        MockChatGenerator("").run([ChatMessage.from_user("hi")], streaming_callback=chunks.append)
        assert chunks[-1].finish_reason == "stop"

    def test_streaming_callback_with_tool_call(self):
        chunks: list[StreamingChunk] = []
        tool_call = ToolCall(tool_name="search", arguments={"query": "x"})
        gen = MockChatGenerator(ChatMessage.from_assistant(tool_calls=[tool_call]))
        gen.run([ChatMessage.from_user("hi")], streaming_callback=chunks.append)
        assert any(chunk.tool_calls for chunk in chunks)
        assert chunks[-1].finish_reason == "tool_calls"

    @pytest.mark.parametrize(
        "generator",
        [
            MockChatGenerator(["a", ChatMessage.from_assistant("b")], model="m", meta={"k": "v"}),
            MockChatGenerator(response_fn=_exclaim),
            MockChatGenerator(),  # echo mode
            MockChatGenerator("hi", streaming_callback=_noop_callback),  # serialized init-level callback
        ],
        ids=["responses", "response_fn", "echo", "streaming_callback"],
    )
    def test_serialization_roundtrip(self, generator):
        restored = MockChatGenerator.from_dict(generator.to_dict())
        assert isinstance(restored, MockChatGenerator)
        # behavior is preserved across the roundtrip
        messages = [ChatMessage.from_user("hi")]
        assert restored.run(messages)["replies"][0].text == generator.run(messages)["replies"][0].text

    def test_in_pipeline(self):
        pipeline = Pipeline()
        pipeline.add_component("generator", MockChatGenerator("from the pipeline"))
        restored = Pipeline.from_dict(pipeline.to_dict())
        result = restored.run({"generator": {"messages": [ChatMessage.from_user("hi")]}})
        assert result["generator"]["replies"][0].text == "from the pipeline"
