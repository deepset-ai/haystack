# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0


import pytest

from haystack import Pipeline
from haystack.components.generators.chat import MockChatGenerator
from haystack.dataclasses import ChatMessage, StreamingChunk, ToolCall


def _exclaim(messages: list[ChatMessage]) -> str:
    """Module-level response function used to test `response_fn` serialization."""
    return f"{messages[-1].text}!"


class TestMockChatGenerator:
    def test_init_default_echo(self):
        gen = MockChatGenerator()
        assert gen._responses is None
        assert gen.response_fn is None
        assert gen.model == "mock-model"
        assert gen.meta == {}

    def test_init_normalizes_string(self):
        gen = MockChatGenerator("hello")
        assert len(gen._responses) == 1
        assert gen._responses[0].text == "hello"
        assert gen._responses[0].role.value == "assistant"

    def test_init_normalizes_list(self):
        gen = MockChatGenerator(["a", ChatMessage.from_assistant("b")])
        assert [msg.text for msg in gen._responses] == ["a", "b"]

    def test_init_rejects_responses_and_response_fn(self):
        with pytest.raises(ValueError, match="either 'responses' or 'response_fn'"):
            MockChatGenerator("a", response_fn=_exclaim)

    def test_init_rejects_empty_list(self):
        with pytest.raises(ValueError, match="must not be an empty list"):
            MockChatGenerator([])

    def test_init_rejects_invalid_response_type(self):
        with pytest.raises(TypeError):
            MockChatGenerator([123])  # type: ignore[list-item]

    def test_fixed_response(self):
        gen = MockChatGenerator("the same answer")
        for _ in range(3):
            result = gen.run([ChatMessage.from_user("anything")])
            assert result["replies"][0].text == "the same answer"

    def test_cycling_responses(self):
        gen = MockChatGenerator(["one", "two", "three"])
        texts = [gen.run([ChatMessage.from_user("hi")])["replies"][0].text for _ in range(4)]
        assert texts == ["one", "two", "three", "one"]

    def test_echo_default_returns_last_user_message(self):
        gen = MockChatGenerator()
        result = gen.run(
            [ChatMessage.from_system("be helpful"), ChatMessage.from_user("first"), ChatMessage.from_user("second")]
        )
        assert result["replies"][0].text == "second"

    def test_echo_default_empty_messages_returns_no_replies(self):
        gen = MockChatGenerator()
        assert gen.run([])["replies"] == []

    def test_response_fn(self):
        gen = MockChatGenerator(response_fn=_exclaim)
        result = gen.run([ChatMessage.from_user("hello")])
        assert result["replies"][0].text == "hello!"

    def test_response_fn_invalid_return_raises(self):
        gen = MockChatGenerator(response_fn=lambda messages: 123)
        with pytest.raises(TypeError, match="must return a string or ChatMessage"):
            gen.run([ChatMessage.from_user("hi")])

    def test_string_input_is_normalized(self):
        gen = MockChatGenerator(response_fn=_exclaim)
        result = gen.run("plain string")
        assert result["replies"][0].text == "plain string!"

    def test_tool_call_response(self):
        tool_call = ToolCall(tool_name="search", arguments={"query": "Haystack"})
        gen = MockChatGenerator(ChatMessage.from_assistant(tool_calls=[tool_call]))
        reply = gen.run([ChatMessage.from_user("search for Haystack")])["replies"][0]
        assert reply.tool_calls == [tool_call]
        assert reply.meta["finish_reason"] == "tool_calls"

    def test_meta_defaults(self):
        gen = MockChatGenerator("hello world")
        meta = gen.run([ChatMessage.from_user("a b c")])["replies"][0].meta
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
        # the stored response must keep its original (empty) meta, untouched by the per-run meta
        assert gen._responses[0].meta == {}

    async def test_run_async(self):
        gen = MockChatGenerator(["one", "two"])
        first = await gen.run_async([ChatMessage.from_user("hi")])
        second = await gen.run_async([ChatMessage.from_user("hi")])
        assert first["replies"][0].text == "one"
        assert second["replies"][0].text == "two"

    def test_streaming_callback_sync(self):
        chunks: list[StreamingChunk] = []
        gen = MockChatGenerator("hello there friend")
        result = gen.run([ChatMessage.from_user("hi")], streaming_callback=chunks.append)
        assert "".join(chunk.content for chunk in chunks) == "hello there friend"
        assert chunks[0].start is True
        assert chunks[-1].finish_reason == "stop"
        # the returned reply matches the predefined response
        assert result["replies"][0].text == "hello there friend"

    async def test_streaming_callback_async(self):
        chunks: list[StreamingChunk] = []

        async def callback(chunk: StreamingChunk) -> None:
            chunks.append(chunk)

        gen = MockChatGenerator("hello world")
        await gen.run_async([ChatMessage.from_user("hi")], streaming_callback=callback)
        assert "".join(chunk.content for chunk in chunks) == "hello world"
        assert chunks[-1].finish_reason == "stop"

    def test_streaming_callback_with_tool_call(self):
        chunks: list[StreamingChunk] = []
        tool_call = ToolCall(tool_name="search", arguments={"query": "x"})
        gen = MockChatGenerator(ChatMessage.from_assistant(tool_calls=[tool_call]))
        gen.run([ChatMessage.from_user("hi")], streaming_callback=chunks.append)
        assert any(chunk.tool_calls for chunk in chunks)
        assert chunks[-1].finish_reason == "tool_calls"

    def test_init_level_streaming_callback(self):
        chunks: list[StreamingChunk] = []
        gen = MockChatGenerator("hello", streaming_callback=chunks.append)
        gen.run([ChatMessage.from_user("hi")])
        assert chunks

    def test_to_dict_from_dict_roundtrip(self):
        gen = MockChatGenerator(["a", ChatMessage.from_assistant("b")], model="m", meta={"k": "v"})
        data = gen.to_dict()
        assert data["type"] == "haystack.components.generators.chat.mock.MockChatGenerator"
        assert data["init_parameters"]["model"] == "m"
        assert data["init_parameters"]["response_fn"] is None

        restored = MockChatGenerator.from_dict(data)
        texts = [restored.run([ChatMessage.from_user("hi")])["replies"][0].text for _ in range(2)]
        assert texts == ["a", "b"]
        assert restored.meta == {"k": "v"}

    def test_to_dict_from_dict_with_response_fn(self):
        gen = MockChatGenerator(response_fn=_exclaim)
        data = gen.to_dict()
        assert data["init_parameters"]["response_fn"].endswith("test_mock._exclaim")
        restored = MockChatGenerator.from_dict(data)
        assert restored.run([ChatMessage.from_user("hello")])["replies"][0].text == "hello!"

    def test_to_dict_echo_mode(self):
        gen = MockChatGenerator()
        data = gen.to_dict()
        assert data["init_parameters"]["responses"] is None
        restored = MockChatGenerator.from_dict(data)
        assert restored.run([ChatMessage.from_user("echo me")])["replies"][0].text == "echo me"

    def test_in_pipeline(self):
        pipeline = Pipeline()
        pipeline.add_component("generator", MockChatGenerator("from the pipeline"))
        result = pipeline.run({"generator": {"messages": [ChatMessage.from_user("hi")]}})
        assert result["generator"]["replies"][0].text == "from the pipeline"

        # the pipeline (and its mock component) survives a serialization roundtrip
        restored = Pipeline.from_dict(pipeline.to_dict())
        result = restored.run({"generator": {"messages": [ChatMessage.from_user("hi")]}})
        assert result["generator"]["replies"][0].text == "from the pipeline"
