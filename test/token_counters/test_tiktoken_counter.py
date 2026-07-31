# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated

import pytest

from haystack.dataclasses import ChatMessage, FileContent, ImageContent, ToolCall
from haystack.token_counters import TiktokenCounter
from haystack.token_counters import tiktoken_counter as tiktoken_counter_module
from haystack.tools import tool

IMAGE = ImageContent(base64_image="Zm9v", mime_type="image/png")
FILE = FileContent(base64_data="Zm9v", mime_type="application/pdf", filename="report.pdf")


def _tool_result(result: str, *, error: bool = False) -> ChatMessage:
    return ChatMessage.from_tool(
        tool_result=result, origin=ToolCall(tool_name="search", arguments={}, id="c1"), error=error
    )


class _FakeEncoder:
    """Stands in for a `tiktoken.Encoding`, recording what it was asked to encode."""

    def __init__(self) -> None:
        self.encoded: list[str] = []

    def encode(self, text: str) -> list[int]:
        self.encoded.append(text)
        return list(range(len(text.split())))


@pytest.fixture
def fake_encoder(monkeypatch) -> _FakeEncoder:
    """Replace `tiktoken.get_encoding`, which downloads its vocabulary on first use."""
    encoder = _FakeEncoder()
    monkeypatch.setattr(tiktoken_counter_module.tiktoken, "get_encoding", lambda _name: encoder)
    return encoder


class TestTiktokenCounter:
    def test_counts_an_empty_conversation_without_an_encoder(self):
        # Short-circuits before the encoder is needed, so it never triggers a download.
        counter = TiktokenCounter()

        assert counter.count([]) == 0
        assert counter._encoder is None

    def test_a_missing_dependency_fails_at_construction(self, monkeypatch):
        # Reported at setup rather than on the first count, which could be many steps into a run.
        def raise_import_error() -> None:
            raise ImportError("Run 'pip install tiktoken'")

        monkeypatch.setattr(tiktoken_counter_module.tiktoken_imports, "check", raise_import_error)

        with pytest.raises(ImportError, match="pip install tiktoken"):
            TiktokenCounter()

    def test_counting_loads_the_encoder_on_demand(self, fake_encoder):
        counter = TiktokenCounter()

        counter.count([ChatMessage.from_user("hi there")])

        assert counter._encoder is fake_encoder

    def test_warm_up_is_idempotent(self, monkeypatch):
        calls: list[str] = []

        def get_encoding(name: str) -> _FakeEncoder:
            calls.append(name)
            return _FakeEncoder()

        monkeypatch.setattr(tiktoken_counter_module.tiktoken, "get_encoding", get_encoding)
        counter = TiktokenCounter(encoding="cl100k_base")

        counter.warm_up()
        counter.warm_up()

        assert calls == ["cl100k_base"]

    def test_encodes_the_rendered_conversation(self, fake_encoder):
        # The counter measures what the renderer produces, so tool calls and non-text parts are represented.
        messages = [
            ChatMessage.from_assistant("looking", tool_calls=[ToolCall(tool_name="search", arguments={"q": "x"})]),
            _tool_result("found it"),
            ChatMessage.from_user(content_parts=["look:", IMAGE, FILE]),
        ]

        TiktokenCounter().count(messages)

        rendered = fake_encoder.encoded[0]
        assert "[assistant] looking" in rendered
        assert '[assistant -> tool_call] search({"q": "x"})' in rendered
        assert "[tool:search] found it" in rendered
        assert "<image>" in rendered
        assert "<file: report.pdf>" in rendered

    def test_serde_round_trip(self):
        data = TiktokenCounter(encoding="cl100k_base", tokens_per_image=200, tokens_per_file=3000).to_dict()

        assert data == {
            "type": "haystack.token_counters.tiktoken_counter.TiktokenCounter",
            "init_parameters": {"encoding": "cl100k_base", "tokens_per_image": 200, "tokens_per_file": 3000},
        }
        restored = TiktokenCounter.from_dict(data)
        assert restored.encoding == "cl100k_base"
        assert restored.tokens_per_image == 200


@tool
def search(query: Annotated[str, "the search query"]) -> str:
    """Search the web for a query and return the top results."""
    return "result"


class TestTiktokenCounterTools:
    def test_tool_schemas_add_to_the_count(self, fake_encoder):
        # A provider is sent the schemas alongside the messages, so they consume tokens too.
        counter = TiktokenCounter()
        messages = [ChatMessage.from_user("hi")]

        assert counter.count(messages, tools=[search]) > counter.count(messages)

    def test_tools_can_be_counted_without_messages(self, fake_encoder):
        assert TiktokenCounter().count([], tools=[search]) > 0

    def test_nothing_to_measure_is_zero(self):
        assert TiktokenCounter().count([]) == 0
        assert TiktokenCounter().count([], tools=None) == 0


@pytest.mark.integration
class TestTiktokenCounterIntegration:
    """Exercises the real encoder, which downloads its vocabulary on first use."""

    def test_counts_grow_with_content(self):
        counter = TiktokenCounter()

        small = counter.count([ChatMessage.from_user("hi")])
        large = counter.count([ChatMessage.from_user("hi " * 500)])

        assert 0 < small < large

    def test_every_message_contributes(self):
        counter = TiktokenCounter()
        messages = [
            ChatMessage.from_system("rules"),
            ChatMessage.from_assistant("looking", tool_calls=[ToolCall(tool_name="search", arguments={"q": "x"})]),
            _tool_result("found it"),
        ]

        assert counter.count(messages) > max(counter.count([message]) for message in messages)

    def test_an_image_is_charged_at_the_flat_rate(self):
        # A tokenizer cannot price an image, so it gets a flat estimate rather than the handful of tokens its
        # placeholder text would cost.
        counter = TiktokenCounter(tokens_per_image=85)

        assert counter.count([ChatMessage.from_user(content_parts=[IMAGE])]) > 85
