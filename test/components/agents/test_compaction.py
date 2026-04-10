# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from unittest.mock import MagicMock

from haystack import component
from haystack.components.agents.compaction import SummarizationCompactionTool
from haystack.components.agents.state import State
from haystack.components.agents.state.state_utils import merge_lists, replace_values
from haystack.dataclasses import ChatMessage, ChatRole
from haystack.tools import Tool


@component
class MockSummarizerGenerator:
    @component.output_types(replies=list[ChatMessage])
    def run(self, messages: list[ChatMessage], **kwargs) -> dict[str, Any]:
        return {"replies": [ChatMessage.from_assistant("Summary of the conversation.")]}


class TestSummarizationCompactionTool:
    def test_init_defaults(self):
        gen = MockSummarizerGenerator()
        tool = SummarizationCompactionTool(chat_generator=gen)

        assert tool.name == "summarization_compaction"
        assert tool.chat_generator is gen
        assert tool.max_messages == 10
        assert tool.summarization_prompt is None
        assert tool.condition is not None
        assert callable(tool.condition)
        assert tool.parameters == {"type": "object", "properties": {}}
        assert tool.inputs_from_state == {"messages": "messages"}
        assert tool.outputs_to_state == {"messages": {"source": "messages", "handler": replace_values}}

    def test_init_custom_params(self):
        gen = MockSummarizerGenerator()
        prompt = "Custom prompt: {messages}"
        tool = SummarizationCompactionTool(chat_generator=gen, max_messages=20, summarization_prompt=prompt)

        assert tool.max_messages == 20
        assert tool.summarization_prompt == prompt

    def test_is_tool_subclass(self):
        gen = MockSummarizerGenerator()
        tool = SummarizationCompactionTool(chat_generator=gen)
        assert isinstance(tool, Tool)

    def test_condition_false_when_below_threshold(self):
        gen = MockSummarizerGenerator()
        tool = SummarizationCompactionTool(chat_generator=gen, max_messages=6)

        state = State(
            schema={"messages": {"type": list[ChatMessage], "handler": merge_lists}},
            data={
                "messages": [
                    ChatMessage.from_user("Hi"),
                    ChatMessage.from_assistant("Hello"),
                    ChatMessage.from_user("How are you?"),
                ]
            },
        )
        assert tool.condition(state) is False

    def test_condition_false_at_exact_threshold(self):
        gen = MockSummarizerGenerator()
        tool = SummarizationCompactionTool(chat_generator=gen, max_messages=4)

        state = State(
            schema={"messages": {"type": list[ChatMessage], "handler": merge_lists}},
            data={
                "messages": [
                    ChatMessage.from_user("a"),
                    ChatMessage.from_assistant("b"),
                    ChatMessage.from_user("c"),
                    ChatMessage.from_assistant("d"),
                ]
            },
        )
        assert tool.condition(state) is False

    def test_condition_true_when_above_threshold(self):
        gen = MockSummarizerGenerator()
        tool = SummarizationCompactionTool(chat_generator=gen, max_messages=4)

        state = State(
            schema={"messages": {"type": list[ChatMessage], "handler": merge_lists}},
            data={
                "messages": [
                    ChatMessage.from_user("a"),
                    ChatMessage.from_assistant("b"),
                    ChatMessage.from_user("c"),
                    ChatMessage.from_assistant("d"),
                    ChatMessage.from_user("e"),
                ]
            },
        )
        assert tool.condition(state) is True

    def test_condition_ignores_system_messages(self):
        gen = MockSummarizerGenerator()
        tool = SummarizationCompactionTool(chat_generator=gen, max_messages=4)

        state = State(
            schema={"messages": {"type": list[ChatMessage], "handler": merge_lists}},
            data={
                "messages": [
                    ChatMessage.from_system("You are helpful"),
                    ChatMessage.from_user("a"),
                    ChatMessage.from_assistant("b"),
                    ChatMessage.from_user("c"),
                    ChatMessage.from_assistant("d"),
                ]
            },
        )
        # 4 non-system messages == max_messages, not above
        assert tool.condition(state) is False

    def test_invoke_summarizes_and_keeps_recent(self):
        gen = MockSummarizerGenerator()
        tool = SummarizationCompactionTool(chat_generator=gen, max_messages=4)

        messages = [
            ChatMessage.from_system("System prompt"),
            ChatMessage.from_user("old msg 1"),
            ChatMessage.from_assistant("old msg 2"),
            ChatMessage.from_user("recent msg 1"),
            ChatMessage.from_assistant("recent msg 2"),
            ChatMessage.from_user("recent msg 3"),
        ]

        result = tool.invoke(messages=messages)

        assert "messages" in result
        new_messages = result["messages"]
        # System message preserved at front
        assert new_messages[0].is_from(ChatRole.SYSTEM)
        assert new_messages[0].text == "System prompt"
        # Summary is second (as a user message)
        assert new_messages[1].is_from(ChatRole.USER)
        assert "[Summary of previous conversation:" in new_messages[1].text
        # keep_count = 4 // 2 = 2, so last 2 non-system messages are kept
        assert new_messages[-1].text == "recent msg 3"
        assert new_messages[-2].text == "recent msg 2"
        assert len(new_messages) == 4  # system + summary + 2 kept

    def test_invoke_with_custom_prompt(self):
        gen = MagicMock()
        gen.run = MagicMock(return_value={"replies": [ChatMessage.from_assistant("custom summary")]})

        custom_prompt = "Please summarize:\n{messages}"
        tool = SummarizationCompactionTool(chat_generator=gen, max_messages=2, summarization_prompt=custom_prompt)

        messages = [ChatMessage.from_user("Hello"), ChatMessage.from_assistant("Hi"), ChatMessage.from_user("Bye")]

        tool.invoke(messages=messages)

        # Verify the custom prompt was used
        call_args = gen.run.call_args
        prompt_msg = call_args.kwargs["messages"][0] if call_args.kwargs else call_args[1]["messages"][0]
        assert prompt_msg.text.startswith("Please summarize:")

    def test_invoke_no_system_messages(self):
        gen = MockSummarizerGenerator()
        tool = SummarizationCompactionTool(chat_generator=gen, max_messages=2)

        messages = [ChatMessage.from_user("old"), ChatMessage.from_assistant("old reply"), ChatMessage.from_user("new")]

        result = tool.invoke(messages=messages)
        new_messages = result["messages"]
        # No system message, so first is summary
        assert new_messages[0].is_from(ChatRole.USER)
        assert "[Summary of previous conversation:" in new_messages[0].text
        # keep_count = 2 // 2 = 1, last 1 non-system kept
        assert new_messages[-1].text == "new"
        assert len(new_messages) == 2  # summary + 1 kept

    def test_to_dict(self):
        gen = MockSummarizerGenerator()
        tool = SummarizationCompactionTool(
            chat_generator=gen, max_messages=15, summarization_prompt="Summarize: {messages}"
        )

        data = tool.to_dict()

        assert data["type"] == "haystack.components.agents.compaction.SummarizationCompactionTool"
        assert data["data"]["max_messages"] == 15
        assert data["data"]["summarization_prompt"] == "Summarize: {messages}"
        assert "chat_generator" in data["data"]
        assert data["data"]["chat_generator"]["type"] == "test_compaction.MockSummarizerGenerator"

    def test_from_dict(self):
        gen = MockSummarizerGenerator()
        tool = SummarizationCompactionTool(chat_generator=gen, max_messages=15)

        data = tool.to_dict()
        restored = SummarizationCompactionTool.from_dict(data)

        assert isinstance(restored, SummarizationCompactionTool)
        assert restored.max_messages == 15
        assert restored.summarization_prompt is None
        assert isinstance(restored.chat_generator, MockSummarizerGenerator)
        assert restored.condition is not None
        assert restored.name == "summarization_compaction"

    def test_serde_roundtrip_preserves_condition_behavior(self):
        gen = MockSummarizerGenerator()
        tool = SummarizationCompactionTool(chat_generator=gen, max_messages=4)

        data = tool.to_dict()
        restored = SummarizationCompactionTool.from_dict(data)

        state = State(
            schema={"messages": {"type": list[ChatMessage], "handler": merge_lists}},
            data={
                "messages": [
                    ChatMessage.from_user("a"),
                    ChatMessage.from_assistant("b"),
                    ChatMessage.from_user("c"),
                    ChatMessage.from_assistant("d"),
                    ChatMessage.from_user("e"),
                ]
            },
        )
        assert restored.condition(state) is True

    def test_empty_parameters_means_auto_invoke(self):
        """Verify the compaction tool has empty parameters, which triggers auto-invoke in the agent."""
        gen = MockSummarizerGenerator()
        tool = SummarizationCompactionTool(chat_generator=gen)
        assert tool.parameters.get("properties") == {}
