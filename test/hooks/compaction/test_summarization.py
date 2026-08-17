# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any, ClassVar

import pytest

from haystack.components.agents import Agent
from haystack.components.generators.chat import MockChatGenerator
from haystack.dataclasses import ChatMessage, ChatRole, FileContent, ImageContent, TextContent, ToolCall
from haystack.hooks.compaction import CompactionHook, SummarizationCompactor
from haystack.hooks.compaction.summarization import _attachment_placeholder
from haystack.hooks.compaction.utils import _COMPACTION_META_KEY
from test.hooks.compaction.helpers import FakeCounter, tool_call, tool_result

pytestmark = pytest.mark.filterwarnings("ignore::haystack.utils.experimental.ExperimentalWarning")

# A target of one token forces every tier to run, isolating the structural rules from sizing.
SMALLEST = 1
# One character per token, so the padded messages below are obviously the expensive ones.
COUNTER = FakeCounter(chars_per_token=1)


def summarizer(*responses: str | Exception) -> tuple[MockChatGenerator, list[str]]:
    """
    A Chat Generator returning the given summaries in order, recording the prompt it received for each.

    An `Exception` among the responses is raised instead of answering, so a test can fail one summarization step.
    """
    queued = list(responses)
    prompts: list[str] = []

    def respond(messages: list[ChatMessage]) -> str:
        prompts.append("\n".join(message.text or "" for message in messages))
        response = queued.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    return MockChatGenerator(response_fn=respond), prompts


class RecordingGenerator(MockChatGenerator):
    """A Chat Generator recording the `generation_kwargs` of every call, and advertising none of its own."""

    def __init__(self) -> None:
        super().__init__("summary")
        self.received_generation_kwargs: list[dict[str, Any] | None] = []

    def run(self, messages, streaming_callback=None, generation_kwargs=None, **kwargs):
        self.received_generation_kwargs.append(generation_kwargs)
        return super().run(messages, streaming_callback, generation_kwargs, **kwargs)

    async def run_async(self, messages, streaming_callback=None, generation_kwargs=None, **kwargs):
        self.received_generation_kwargs.append(generation_kwargs)
        return await super().run_async(messages, streaming_callback, generation_kwargs, **kwargs)


class MappedGenerator(RecordingGenerator):
    """A Chat Generator naming what its provider calls Haystack's `max_output_tokens`."""

    _HAYSTACK_TO_PROVIDER_GENERATION_KWARGS: ClassVar[dict[str, str]] = {"max_output_tokens": "provider_max_tokens"}


def summary(text: str, source: str) -> ChatMessage:
    """A summary an earlier compaction left behind, marked the way this compactor marks its own."""
    return ChatMessage.from_user(
        f"<conversation_summary>\n{text}\n</conversation_summary>",
        meta={_COMPACTION_META_KEY: {"strategy": "summarization", "source": source}},
    )


def sources(messages: list[ChatMessage]) -> list[str]:
    """Which stretch of conversation each summary in `messages` stands in for, oldest first."""
    return [
        message.meta[_COMPACTION_META_KEY]["source"] for message in messages if _COMPACTION_META_KEY in message.meta
    ]


def two_turns_and_a_task() -> list[ChatMessage]:
    """A padded oldest turn, a short recent turn, and the current task with one step behind it."""
    return [
        ChatMessage.from_system("rules"),
        ChatMessage.from_user("oldest question " * 30),
        ChatMessage.from_assistant("oldest answer " * 30),
        ChatMessage.from_user("recent question"),
        ChatMessage.from_assistant("recent answer"),
        ChatMessage.from_user("current task"),
        ChatMessage.from_assistant("current step"),
    ]


def a_task_with_two_steps() -> list[ChatMessage]:
    """The current task with a padded oldest step and a cheap newest one, and no history in front of it."""
    return [
        ChatMessage.from_system("rules"),
        ChatMessage.from_user("current task"),
        tool_call("old"),
        tool_result("old result " * 30, call_id="old"),
        tool_call("new"),
        tool_result("new result", call_id="new"),
    ]


class TestAttachmentPlaceholder:
    @pytest.mark.parametrize(
        ("content", "expected"),
        [
            pytest.param(ImageContent(base64_image="Zm9v", mime_type="image/png"), "<image: image/png>", id="image"),
            pytest.param(
                ImageContent(base64_image="Zm9v", mime_type="image/png", meta={"file_path": "/tmp/shot.png"}),
                "<image: image/png, file_path=/tmp/shot.png>",
                id="image-named-by-meta",
            ),
            pytest.param(
                FileContent(base64_data="Zm9v", mime_type="application/pdf", filename="q3.pdf"),
                "<file: q3.pdf, application/pdf>",
                id="file",
            ),
            pytest.param(
                FileContent(base64_data="Zm9v", mime_type="application/pdf", extra={"page": 4}),
                "<file: unnamed, application/pdf, page=4>",
                id="file-unnamed-with-extra",
            ),
            # A nested value could be arbitrarily large, so it is left out rather than bloating the prompt.
            pytest.param(
                ImageContent(base64_image="Zm9v", mime_type="image/png", meta={"boxes": [[1, 2], [3, 4]]}),
                "<image: image/png>",
                id="nested-metadata-left-out",
            ),
        ],
    )
    def test_names_the_attachment(self, content, expected):
        assert _attachment_placeholder(content) == expected


class TestSummarizationCompactor:
    def test_summarizes_oldest_historical_turn(self):
        messages = two_turns_and_a_task()
        generator, prompts = summarizer("short historical summary")
        # Room for everything but the padded oldest turn, plus the summary standing in for it.
        target_tokens = COUNTER.count([messages[0], *messages[3:]]) + 100

        compacted = SummarizationCompactor(generator, max_summary_tokens=100).compact(
            messages=messages, target_tokens=target_tokens, token_counter=COUNTER
        )

        assert compacted is not None
        # Only the oldest turn was summarized, so the recent turn never reached the generator.
        assert len(prompts) == 1
        assert "oldest question" in prompts[0]
        assert "recent question" not in prompts[0]
        assert compacted == [messages[0], compacted[1], *messages[3:]]

    def test_leaves_the_input_conversation_untouched(self):
        messages = two_turns_and_a_task()
        SummarizationCompactor(MockChatGenerator("summary"), max_summary_tokens=1).compact(
            messages=messages, target_tokens=SMALLEST, token_counter=COUNTER
        )
        assert messages == two_turns_and_a_task()

    def test_summarizes_historical_turns_then_current_steps(self):
        messages = [
            ChatMessage.from_system("rules"),
            ChatMessage.from_user("old question " * 40),
            ChatMessage.from_assistant("old answer " * 40),
            *a_task_with_two_steps()[1:],
        ]
        generator, prompts = summarizer("history", "old step")

        compacted = SummarizationCompactor(generator, max_summary_tokens=1).compact(
            messages=messages, target_tokens=SMALLEST, token_counter=COUNTER
        )

        assert compacted is not None
        assert "old question" in prompts[0]
        assert "old result" in prompts[1]
        assert sources(compacted) == ["historical_turns", "current_task_steps"]
        # The newest step is never given up.
        assert compacted[-2:] == messages[-2:]

    def test_folds_historical_summaries_before_current_steps(self):
        messages = [
            ChatMessage.from_system("rules"),
            summary("first history " * 20, "historical_turns"),
            summary("second history " * 20, "historical_turns"),
            *a_task_with_two_steps()[1:],
        ]
        generator, prompts = summarizer("combined history", "old step")

        compacted = SummarizationCompactor(generator, max_summary_tokens=1).compact(
            messages=messages, target_tokens=SMALLEST, token_counter=COUNTER
        )

        assert compacted is not None
        # The two historical summaries are folded into one before any current-task step is touched.
        assert "first history" in prompts[0] and "old result" not in prompts[0]
        assert "old result" in prompts[1]
        assert sources(compacted) == ["historical_summaries", "current_task_steps"]

    def test_folds_current_task_summaries_before_more_steps(self):
        messages = [
            ChatMessage.from_system("rules"),
            ChatMessage.from_user("current task"),
            summary("first step summary " * 20, "current_task_steps"),
            summary("second step summary " * 20, "current_task_steps"),
            *a_task_with_two_steps()[2:],
        ]
        generator, prompts = summarizer("combined steps", "old step")

        compacted = SummarizationCompactor(generator, max_summary_tokens=1).compact(
            messages=messages, target_tokens=SMALLEST, token_counter=COUNTER
        )

        assert compacted is not None
        assert "first step summary" in prompts[0]
        assert "old result" in prompts[1]
        assert sources(compacted) == ["current_task_summaries", "current_task_steps"]
        assert compacted[-2:] == messages[-2:]

    @pytest.mark.parametrize(("min_keep_steps", "expected"), [(0, 0), (1, 1), (2, 2), (20, 2)])
    def test_min_keep_steps_wins_over_an_unaffordable_target(self, min_keep_steps, expected):
        messages = a_task_with_two_steps()
        generator, _ = summarizer("step summary")
        compacted = SummarizationCompactor(generator, min_keep_steps=min_keep_steps, max_summary_tokens=1).compact(
            messages=messages, target_tokens=SMALLEST, token_counter=COUNTER
        )
        result = compacted or messages
        assert sum(message.is_from(role=ChatRole.ASSISTANT) for message in result) == expected

    def test_attachments_are_named_in_the_transcript(self):
        image = ImageContent(base64_image="Zm9v", mime_type="image/png", meta={"file_path": "/tmp/shot.png"})
        pdf = FileContent(base64_data="Zm9v", mime_type="application/pdf", filename="q3.pdf")
        messages = [
            ChatMessage.from_system("rules"),
            ChatMessage.from_user(content_parts=["review this " * 20, pdf]),
            tool_call("c1"),
            # An attachment a tool returned is nested inside the tool result rather than on the message.
            ChatMessage.from_tool(
                tool_result=[TextContent(text="captured " * 20), image],
                origin=ToolCall(tool_name="browse", arguments={}, id="c1"),
            ),
            ChatMessage.from_user("current task"),
        ]
        generator, prompts = summarizer("summary")
        SummarizationCompactor(generator, max_summary_tokens=1).compact(
            messages=messages, target_tokens=SMALLEST, token_counter=COUNTER
        )
        # The summary cannot reproduce either attachment, so the transcript has to name them well enough to ask again.
        assert "<file: q3.pdf, application/pdf>" in prompts[0]
        assert "<image: image/png, file_path=/tmp/shot.png>" in prompts[0]

    def test_custom_summary_instruction_replaces_the_default(self):
        generator, prompts = summarizer("summary")
        SummarizationCompactor(generator, summary_instruction="Only list file paths.", max_summary_tokens=1).compact(
            messages=two_turns_and_a_task(), target_tokens=SMALLEST, token_counter=COUNTER
        )
        assert "Only list file paths." in prompts[0]
        assert "You are compacting part of a conversation" not in prompts[0]

    def test_keeps_partial_progress_when_a_summary_fails(self):
        messages = [
            ChatMessage.from_system("rules"),
            ChatMessage.from_user("old question " * 30),
            ChatMessage.from_assistant("old answer " * 30),
            ChatMessage.from_user("current task"),
            ChatMessage.from_assistant("old step " * 30),
            ChatMessage.from_assistant("new step"),
        ]
        generator, prompts = summarizer("history", RuntimeError("provider unavailable"))

        compacted = SummarizationCompactor(generator, max_summary_tokens=1).compact(
            messages=messages, target_tokens=SMALLEST, token_counter=COUNTER
        )

        assert compacted is not None
        assert len(prompts) == 2
        # The history was summarized before the step summary failed, and that progress is kept.
        assert sources(compacted) == ["historical_turns"]
        assert compacted[-2:] == messages[-2:]

    def test_raises_when_a_summary_does_not_shrink_the_conversation(self):
        messages = [
            ChatMessage.from_system("rules"),
            ChatMessage.from_user("old"),
            ChatMessage.from_assistant("answer"),
            ChatMessage.from_user("current"),
        ]
        compactor = SummarizationCompactor(
            MockChatGenerator("much longer summary " * 100), max_summary_tokens=1, raise_on_failure=True
        )
        with pytest.raises(RuntimeError, match="did not reduce"):
            compactor.compact(messages=messages, target_tokens=SMALLEST, token_counter=COUNTER)

    def test_returns_none_when_the_conversation_fits(self):
        generator, prompts = summarizer("unused")
        messages = [ChatMessage.from_system("rules"), ChatMessage.from_user("task")]
        compacted = SummarizationCompactor(generator).compact(
            messages=messages, target_tokens=10_000, token_counter=COUNTER
        )
        assert compacted is None
        assert prompts == []

    def test_summary_is_a_marked_user_message(self):
        messages = [
            ChatMessage.from_system("rules"),
            ChatMessage.from_user("old " * 100),
            ChatMessage.from_assistant("answer " * 100),
            ChatMessage.from_user("task"),
        ]
        compacted = SummarizationCompactor(MockChatGenerator("summary"), max_summary_tokens=1).compact(
            messages=messages, target_tokens=SMALLEST, token_counter=COUNTER
        )
        assert compacted is not None
        assert compacted[1].is_from(role=ChatRole.USER)
        assert compacted[1].meta[_COMPACTION_META_KEY] == {
            "strategy": "summarization",
            "summarized_messages": 2,
            "source": "historical_turns",
        }

    @pytest.mark.parametrize(
        ("generator_class", "expected"),
        [
            # A generator that maps `max_output_tokens` is held to the budget by its own provider setting.
            pytest.param(MappedGenerator, {"provider_max_tokens": 64}, id="advertised"),
            # Nothing is guessed for a generator that maps nothing, so it keeps whatever it was configured with.
            pytest.param(RecordingGenerator, None, id="not-advertised"),
        ],
    )
    def test_summary_budget_is_sent_only_when_the_generator_supports_a_limit(self, generator_class, expected):
        generator = generator_class()

        SummarizationCompactor(generator, max_summary_tokens=64).compact(
            messages=two_turns_and_a_task(), target_tokens=SMALLEST, token_counter=COUNTER
        )

        assert generator.received_generation_kwargs
        assert all(received == expected for received in generator.received_generation_kwargs)

    def test_summary_budget_is_always_stated_in_the_prompt(self):
        generator, prompts = summarizer("summary")
        SummarizationCompactor(generator, max_summary_tokens=64).compact(
            messages=two_turns_and_a_task(), target_tokens=SMALLEST, token_counter=COUNTER
        )
        assert "no more than approximately 64 tokens" in prompts[0]

    def test_warns_when_the_generator_cannot_enforce_the_summary_budget(self, caplog):
        with caplog.at_level(logging.WARNING):
            SummarizationCompactor(RecordingGenerator(), max_summary_tokens=64)
        assert "does not advertise a generation-parameter mapping for `max_output_tokens`" in caplog.text
        assert "set its provider-specific output-token limit to 64" in caplog.text

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"min_keep_steps": -1}, "`min_keep_steps` must be at least 0"),
            ({"max_summary_tokens": 0}, "`max_summary_tokens` must be a positive"),
        ],
    )
    def test_rejects_invalid_settings(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            SummarizationCompactor(MockChatGenerator("summary"), **kwargs)

    def test_serde_round_trip(self):
        compactor = SummarizationCompactor(
            MockChatGenerator("summary"),
            min_keep_steps=2,
            max_summary_tokens=321,
            summary_instruction="custom",
            raise_on_failure=True,
        )
        restored = SummarizationCompactor.from_dict(compactor.to_dict())
        assert isinstance(restored.chat_generator, MockChatGenerator)
        assert restored.min_keep_steps == 2
        assert restored.max_summary_tokens == 321
        assert restored.summary_instruction == "custom"
        assert restored.raise_on_failure is True


class TestSummarizationCompactorInAgent:
    def test_compacts_history_through_a_compaction_hook(self):
        summary_generator = MappedGenerator()
        hook = CompactionHook(
            compactor=SummarizationCompactor(summary_generator, max_summary_tokens=64),
            context_window=1_000,
            compact_at=0.5,
            compact_to=0.2,
            token_counter=COUNTER,
        )
        agent = Agent(chat_generator=MockChatGenerator("done"), system_prompt="rules", hooks={"before_llm": [hook]})
        messages = [
            ChatMessage.from_user("old question " * 30),
            ChatMessage.from_assistant("old answer " * 30),
            ChatMessage.from_user("current task"),
        ]
        result = agent.run(messages=messages)
        compacted = result["messages"]
        assert result["last_message"].text == "done"
        assert compacted[0].text == "rules"
        assert any(message.text == "current task" for message in compacted)
        assert sources(compacted) == ["historical_turns"]
        assert all("old question" not in (message.text or "") for message in compacted)
        assert summary_generator.received_generation_kwargs == [{"provider_max_tokens": 64}]


class TestSummarizationCompactorAsync:
    @pytest.mark.asyncio
    async def test_compact_async_matches_compact(self):
        messages = two_turns_and_a_task()
        generator, prompts = summarizer("async summary")

        compacted = await SummarizationCompactor(generator, max_summary_tokens=1).compact_async(
            messages=messages, target_tokens=SMALLEST, token_counter=COUNTER
        )

        assert len(prompts) == 1
        assert compacted == SummarizationCompactor(MockChatGenerator("async summary"), max_summary_tokens=1).compact(
            messages=messages, target_tokens=SMALLEST, token_counter=COUNTER
        )
