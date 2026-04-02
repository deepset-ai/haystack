# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any

from haystack.components.agents.state.state_utils import replace_values
from haystack.core.serialization import component_from_dict, component_to_dict, generate_qualified_class_name
from haystack.dataclasses import ChatMessage, ChatRole
from haystack.tools import Tool

if TYPE_CHECKING:
    from haystack.components.agents.state import State
    from haystack.components.generators.chat.types import ChatGenerator


_COMPACTION_PARAMETERS: dict[str, Any] = {"type": "object", "properties": {}}


class SummarizationCompactionTool(Tool):
    """
    A tool that summarizes the oldest messages once the history exceeds a length threshold.

    The condition fires before each LLM call. When the number of non-system messages exceeds
    `max_messages`, the older portion is summarized by `chat_generator` into a single user message.
    The most recent `max_messages // 2` non-system messages are kept verbatim so the LLM retains
    immediate context. The system message (if any) is always preserved at the front.

    Because this tool is condition-triggered it is never exposed to the LLM and cannot be called
    by the model directly.

    Usage example:
    ```python
    from haystack.components.agents import Agent
    from haystack.components.agents.compaction import SummarizationCompactionTool
    from haystack.components.generators.chat import OpenAIChatGenerator

    compaction = SummarizationCompactionTool(
        chat_generator=OpenAIChatGenerator(),
        max_messages=20,
    )
    agent = Agent(chat_generator=OpenAIChatGenerator(), tools=[compaction, ...])
    ```
    """

    def __init__(
        self, chat_generator: "ChatGenerator", max_messages: int = 10, summarization_prompt: str | None = None
    ) -> None:
        """
        Initialize the SummarizationCompactionTool.

        :param chat_generator: A chat generator used to produce the summary. Called synchronously.
        :param max_messages: Number of non-system messages that triggers compaction. Defaults to 10.
        :param summarization_prompt: Prompt sent to the chat generator to request a summary. The
            placeholder ``{messages}`` is replaced with the formatted conversation excerpt to summarize.
            If not provided, a sensible default is used.
        """
        self.chat_generator = chat_generator
        self.max_messages = max_messages
        self.summarization_prompt = summarization_prompt

        _prompt = summarization_prompt or (
            "Summarize the following conversation concisely, preserving all important context, "
            "decisions, and information that would be needed to continue the conversation:\n\n{messages}"
        )

        def _condition(state: "State") -> bool:
            messages: list[ChatMessage] = state.get("messages") or []
            non_system = [m for m in messages if not m.is_from(ChatRole.SYSTEM)]
            return len(non_system) > max_messages

        def _summarize(messages: list[ChatMessage]) -> dict[str, Any]:
            system_msgs = [m for m in messages if m.is_from(ChatRole.SYSTEM)]
            non_system = [m for m in messages if not m.is_from(ChatRole.SYSTEM)]

            keep_count = max_messages // 2
            to_summarize = non_system[:-keep_count] if keep_count else non_system
            to_keep = non_system[-keep_count:] if keep_count else []

            messages_text = "\n".join(f"{m.role.value}: {m.text}" for m in to_summarize if m.text)
            prompt = _prompt.format(messages=messages_text)

            result = chat_generator.run(messages=[ChatMessage.from_user(prompt)])
            summary_text = result["replies"][0].text or ""

            summary_msg = ChatMessage.from_user(f"[Summary of previous conversation: {summary_text}]")
            return {"messages": system_msgs + [summary_msg] + to_keep}

        super().__init__(
            name="summarization_compaction",
            description="Summarizes the oldest chat messages to reduce context length.",
            parameters=_COMPACTION_PARAMETERS,
            function=_summarize,
            inputs_from_state={"messages": "messages"},
            outputs_to_state={"messages": {"source": "messages", "handler": replace_values}},
            condition=_condition,
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the tool to a dictionary.

        :returns: Dictionary with serialized data.
        """
        serialized: dict[str, Any] = {
            "chat_generator": component_to_dict(obj=self.chat_generator, name="chat_generator"),
            "max_messages": self.max_messages,
            "summarization_prompt": self.summarization_prompt,
        }
        return {"type": generate_qualified_class_name(type(self)), "data": serialized}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SummarizationCompactionTool":
        """
        Deserializes the tool from a dictionary.

        :param data: Dictionary to deserialize from.
        :returns: Deserialized tool.
        """
        from haystack.core.serialization import import_class_by_name

        inner_data = data["data"]
        generator_data = inner_data["chat_generator"]
        generator_class = import_class_by_name(generator_data["type"])
        inner_data["chat_generator"] = component_from_dict(
            cls=generator_class, data=generator_data, name="chat_generator"
        )
        return cls(**inner_data)
