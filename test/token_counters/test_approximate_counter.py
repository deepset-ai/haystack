# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated

import pytest

from haystack.dataclasses import ChatMessage, FileContent, ImageContent, TextContent, ToolCall
from haystack.token_counters import ApproximateTokenCounter
from haystack.token_counters.utils import _rendered_conversation
from haystack.tools import tool

IMAGE = ImageContent(base64_image="Zm9v", mime_type="image/png")
FILE = FileContent(base64_data="Zm9v", mime_type="application/pdf", filename="report.pdf")


class TestApproximateTokenCounter:
    def test_counts_an_empty_conversation_as_zero(self):
        assert ApproximateTokenCounter().count([]) == 0

    def test_counts_at_the_configured_ratio(self):
        messages = [ChatMessage.from_user("x" * 400)]
        rendered = len(_rendered_conversation(messages))

        assert ApproximateTokenCounter().count(messages) == rendered // 4
        assert ApproximateTokenCounter(chars_per_token=2).count(messages) == rendered // 2

    def test_needs_no_dependency_or_warm_up(self):
        # The whole point of this counter: it works straight away, with nothing installed and nothing loaded.
        assert ApproximateTokenCounter().count([ChatMessage.from_user("hi")]) > 0

    def test_counts_grow_with_content(self):
        counter = ApproximateTokenCounter()

        assert counter.count([ChatMessage.from_user("hi")]) < counter.count([ChatMessage.from_user("hi " * 500)])

    @pytest.mark.parametrize(
        ("content_parts", "expected_flat"),
        [
            pytest.param([IMAGE], 85, id="image"),
            pytest.param([FILE], 1000, id="file"),
            pytest.param([IMAGE, IMAGE, FILE], 85 * 2 + 1000, id="several"),
        ],
    )
    def test_non_text_content_is_charged_at_a_flat_rate(self, content_parts, expected_flat):
        # Neither has text to measure, so each gets a flat estimate on top of whatever its placeholder renders to.
        count = ApproximateTokenCounter().count([ChatMessage.from_user(content_parts=content_parts)])

        assert count > expected_flat

    def test_an_image_inside_a_tool_result_is_counted(self):
        # `ChatMessage.images` does not see these, but a tool returning a screenshot puts them here, so a counter
        # looking only at a message's own content would miss them entirely.
        counter = ApproximateTokenCounter()
        nested = ChatMessage.from_tool(
            tool_result=[TextContent(text="shot:"), IMAGE], origin=ToolCall(tool_name="shot", arguments={}, id="c1")
        )

        assert nested.images == []
        assert counter.count([nested]) > 85

    def test_the_flat_rates_are_configurable(self):
        messages = [ChatMessage.from_user(content_parts=[IMAGE])]

        cheap = ApproximateTokenCounter(tokens_per_image=10).count(messages)
        dear = ApproximateTokenCounter(tokens_per_image=500).count(messages)

        assert dear - cheap == 490

    def test_rejects_a_non_positive_ratio(self):
        with pytest.raises(ValueError, match="`chars_per_token` must be greater than 0"):
            ApproximateTokenCounter(chars_per_token=0)

    def test_serde_round_trip(self):
        data = ApproximateTokenCounter(chars_per_token=3.5, tokens_per_image=200, tokens_per_file=3000).to_dict()

        assert data == {
            "type": "haystack.token_counters.approximate_counter.ApproximateTokenCounter",
            "init_parameters": {"chars_per_token": 3.5, "tokens_per_image": 200, "tokens_per_file": 3000},
        }
        restored = ApproximateTokenCounter.from_dict(data)
        assert restored.chars_per_token == 3.5
        assert restored.tokens_per_file == 3000


@tool
def search(query: Annotated[str, "the search query"]) -> str:
    """Search the web for a query and return the top results."""
    return "result"


class TestApproximateTokenCounterTools:
    def test_tool_schemas_add_to_the_count(self):
        # A provider is sent the schemas alongside the messages, so they consume tokens too.
        counter = ApproximateTokenCounter()
        messages = [ChatMessage.from_user("hi")]

        assert counter.count(messages, tools=[search]) > counter.count(messages)

    def test_tools_can_be_counted_without_messages(self):
        assert ApproximateTokenCounter().count([], tools=[search]) > 0

    def test_nothing_to_measure_is_zero(self):
        assert ApproximateTokenCounter().count([]) == 0
        assert ApproximateTokenCounter().count([], tools=None) == 0
