# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0


import pytest

from haystack.components.agents import Agent
from haystack.components.generators.chat import MockChatGenerator
from haystack.dataclasses import ChatMessage, FileContent, ImageContent, TextContent, ToolCall
from haystack.hooks.compaction import CompactionHook, SummarizationCompactor
from haystack.hooks.compaction.summarization import _attachment_placeholder
from haystack.hooks.compaction.utils import _COMPACTION_META_KEY
from test.hooks.compaction.helpers import FakeCounter, fresh_conversation_with_two_steps, tool_call, tool_result

pytestmark = pytest.mark.filterwarnings("ignore::haystack.utils.experimental.ExperimentalWarning")

# One character per token, so the padded messages below are obviously the expensive ones.
COUNTER = FakeCounter(chars_per_token=1)


def summarizer(*responses: str | Exception) -> tuple[MockChatGenerator, list[str]]:
    """
    A Chat Generator returning the given summaries in order, recording the prompt it received for each.

    An `Exception` among the responses is raised instead of answering, so a test can fail one summarization step.
    Calling it more often than there are responses raises, so a test that queues none asserts nothing was summarized.
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


class NoReplyGenerator(MockChatGenerator):
    """A Chat Generator answering with no replies at all, as a misbehaving provider or proxy can."""

    def __init__(self) -> None:
        super().__init__("unused")

    def run(self, messages, streaming_callback=None, generation_kwargs=None, **kwargs):
        return {"replies": []}

    async def run_async(self, messages, streaming_callback=None, generation_kwargs=None, **kwargs):
        return {"replies": []}


def summary(text: str, source: str, summarized_messages: int = 1) -> ChatMessage:
    """Build a summary message with the metadata written by the compactor."""
    return ChatMessage.from_user(
        f"<conversation_summary>\n{text}\n</conversation_summary>",
        meta={
            _COMPACTION_META_KEY: {
                "strategy": "summarization",
                "summarized_messages": summarized_messages,
                "source": source,
            }
        },
    )


def summaries(messages: list[ChatMessage]) -> list[ChatMessage]:
    """Return the summary messages in a conversation, oldest first."""
    return [message for message in messages if _COMPACTION_META_KEY in message.meta]


def compact_after_each_addition(
    compactor: SummarizationCompactor,
    initial_messages: list[ChatMessage],
    additions: list[list[ChatMessage]],
    target_tokens: int,
) -> list[list[ChatMessage]]:
    """
    Grow and compact a conversation the way an Agent loop does, preserving the state after each addition.

    The snapshots make behavior across separate `compact` calls visible to lifecycle tests.
    """
    messages = initial_messages
    snapshots = []
    for addition in additions:
        messages = [*messages, *addition]
        compacted = compactor.compact(messages=messages, target_tokens=target_tokens, token_counter=COUNTER)
        messages = compacted if compacted is not None else messages
        snapshots.append(messages)
    return snapshots


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


class TestNextSummarySelection:
    def test_selects_the_fewest_oldest_historical_turns(self):
        # Two completed historical turns followed by a current task with one step.
        messages = [
            ChatMessage.from_system("rules"),
            ChatMessage.from_user("oldest question " * 30),
            ChatMessage.from_assistant("oldest answer " * 30),
            ChatMessage.from_user("recent question"),
            ChatMessage.from_assistant("recent answer"),
            ChatMessage.from_user("current task"),
            ChatMessage.from_assistant("current step"),
        ]
        # Room for everything but the padded oldest turn, plus the summary standing in for it.
        target_tokens = COUNTER.count([messages[0], *messages[3:]]) + 100
        plan = SummarizationCompactor(chat_generator=MockChatGenerator(), approximate_summary_tokens=100)._next_summary(
            messages=messages, target_tokens=target_tokens, token_counter=COUNTER
        )
        assert plan == ([1, 2], "historical_turns")

    def test_selects_historical_turns_before_current_steps(self):
        messages = [
            ChatMessage.from_system("rules"),
            ChatMessage.from_user("old question " * 40),
            ChatMessage.from_assistant("old answer " * 40),
            *fresh_conversation_with_two_steps()[1:],
        ]
        plan = SummarizationCompactor(chat_generator=MockChatGenerator(), approximate_summary_tokens=1)._next_summary(
            messages=messages, target_tokens=1, token_counter=COUNTER
        )
        assert plan == ([1, 2], "historical_turns")

    def test_selects_historical_summaries_before_current_steps(self):
        messages = [
            ChatMessage.from_system("rules"),
            summary(text="first history " * 20, source="historical_turns"),
            summary(text="second history " * 20, source="historical_turns"),
            *fresh_conversation_with_two_steps()[1:],
        ]
        plan = SummarizationCompactor(chat_generator=MockChatGenerator(), approximate_summary_tokens=1)._next_summary(
            messages=messages, target_tokens=1, token_counter=COUNTER
        )
        assert plan == ([1, 2], "historical_summaries")

    def test_selects_current_steps_before_current_task_summaries(self):
        messages = [
            ChatMessage.from_system("rules"),
            ChatMessage.from_user("current task"),
            summary(text="first step summary " * 20, source="current_task_steps"),
            summary(text="second step summary " * 20, source="current_task_steps"),
            *fresh_conversation_with_two_steps()[2:],
        ]
        plan = SummarizationCompactor(chat_generator=MockChatGenerator(), approximate_summary_tokens=1)._next_summary(
            messages=messages, target_tokens=1, token_counter=COUNTER
        )
        assert plan == ([4, 5], "current_task_steps")

    def test_selects_current_task_summaries_when_min_keep_steps_reserves_every_step(self):
        messages = [
            ChatMessage.from_system("rules"),
            ChatMessage.from_user("current task"),
            summary(text="first step summary " * 20, source="current_task_steps"),
            summary(text="second step summary " * 20, source="current_task_steps"),
            tool_call("new"),
            tool_result("new result", call_id="new"),
        ]
        plan = SummarizationCompactor(
            chat_generator=MockChatGenerator(), min_keep_steps=1, approximate_summary_tokens=1
        )._next_summary(messages=messages, target_tokens=1, token_counter=COUNTER)
        assert plan == ([2, 3], "current_task_summaries")

    @pytest.mark.parametrize(
        ("min_keep_steps", "expected"),
        [
            pytest.param(0, ([2, 3, 4, 5], "current_task_steps"), id="keep-none"),
            pytest.param(1, ([2, 3], "current_task_steps"), id="keep-one"),
            pytest.param(2, None, id="keep-both"),
            pytest.param(20, None, id="keep-more-than-exist"),
        ],
    )
    def test_min_keep_steps_limits_eligible_current_steps(self, min_keep_steps, expected):
        plan = SummarizationCompactor(
            chat_generator=MockChatGenerator(), min_keep_steps=min_keep_steps, approximate_summary_tokens=1
        )._next_summary(messages=fresh_conversation_with_two_steps(), target_tokens=1, token_counter=COUNTER)
        assert plan == expected

    def test_returns_none_at_the_compaction_floor(self):
        # Nothing may be removed once each region has one summary and the newest step is reserved.
        messages = [
            ChatMessage.from_system("rules"),
            summary(text="all history " * 20, source="historical_summaries"),
            ChatMessage.from_user("current task"),
            summary(text="all earlier steps " * 20, source="current_task_summaries"),
            tool_call("new"),
            tool_result("new result " * 20, call_id="new"),
        ]
        plan = SummarizationCompactor(
            chat_generator=MockChatGenerator(), min_keep_steps=1, approximate_summary_tokens=1
        )._next_summary(messages=messages, target_tokens=1, token_counter=COUNTER)
        assert plan is None


class TestCompaction:
    def test_does_not_mutate_the_input(self):
        messages = fresh_conversation_with_two_steps()
        SummarizationCompactor(MockChatGenerator("summary"), approximate_summary_tokens=1).compact(
            messages=messages, target_tokens=1, token_counter=COUNTER
        )
        assert messages == fresh_conversation_with_two_steps()

    def test_returns_none_when_the_conversation_fits(self):
        generator, prompts = summarizer()
        messages = [ChatMessage.from_system("rules"), ChatMessage.from_user("task")]
        compacted = SummarizationCompactor(generator).compact(
            messages=messages, target_tokens=10_000, token_counter=COUNTER
        )
        assert compacted is None
        assert prompts == []

    def test_historical_summaries_accumulate_while_raw_turns_remain(self):
        compactor = SummarizationCompactor(MockChatGenerator("summary"), approximate_summary_tokens=10)
        # Every new user message moves the preceding turn into history. The target lets each `compact` call summarize
        # only that new historical turn, leaving earlier summaries separate while raw turns remain.
        additions = [
            [ChatMessage.from_user(f"question {index} " * 30), ChatMessage.from_assistant(f"answer {index} " * 30)]
            for index in range(4)
        ]
        snapshots = compact_after_each_addition(
            compactor=compactor,
            initial_messages=[ChatMessage.from_system("rules")],
            additions=additions,
            target_tokens=1_100,
        )
        expected_summary = summary(text="summary", source="historical_turns", summarized_messages=2)
        summaries_per_snapshot = [summaries(messages=snapshot) for snapshot in snapshots]
        assert summaries_per_snapshot == [
            [],
            [expected_summary],
            [expected_summary, expected_summary],
            [expected_summary, expected_summary, expected_summary],
        ]

    def test_current_task_summaries_accumulate_while_raw_steps_remain(self):
        compactor = SummarizationCompactor(
            MockChatGenerator("summary"), min_keep_steps=1, approximate_summary_tokens=10
        )
        # Every addition completes another Agent step. The newest step remains reserved, so each `compact` call
        # summarizes the step that just became eligible and leaves earlier summaries separate.
        additions = [
            [tool_call(f"c{index}"), tool_result(f"result {index} " * 30, call_id=f"c{index}")] for index in range(4)
        ]
        snapshots = compact_after_each_addition(
            compactor=compactor,
            initial_messages=[ChatMessage.from_system("rules"), ChatMessage.from_user("current task")],
            additions=additions,
            target_tokens=650,
        )
        expected_summary = summary(text="summary", source="current_task_steps", summarized_messages=2)
        summaries_per_snapshot = [summaries(messages=snapshot) for snapshot in snapshots]
        assert summaries_per_snapshot == [
            [],
            [expected_summary],
            [expected_summary, expected_summary],
            [expected_summary, expected_summary, expected_summary],
        ]

    def test_a_summarized_past_task_is_combined_into_history_once_a_new_task_arrives(self):
        # The recorded source describes how a summary was created; its position decides which region it now occupies.
        messages = [
            ChatMessage.from_system("rules"),
            ChatMessage.from_user("past task " * 30),
            summary(text="early steps of the past task", source="current_task_steps"),
            ChatMessage.from_assistant("late step " * 30),
            ChatMessage.from_user("current task"),
        ]
        generator, prompts = summarizer("remaining past-task messages", "combined history")
        compacted = SummarizationCompactor(generator, approximate_summary_tokens=5).compact(
            messages=messages, target_tokens=1, token_counter=COUNTER
        )
        assert compacted is not None
        assert len(prompts) == 2
        # TODO I feel a little concerned about this. Does this mean message 1 and 3 were summarized together, leaving
        #      message 2 alone? And then the two summaries were combined? Seems like the wrong order and unnecessary
        #      amounts of llm calls.
        # First, only the raw messages from the past task are summarized; its existing summary is left in place.
        assert "past task" in prompts[0] and "early steps of the past task" not in prompts[0]
        # Then the two historical summaries are combined.
        assert "early steps of the past task" in prompts[1] and "remaining past-task messages" in prompts[1]
        assert summaries(messages=compacted) == [
            summary(text="combined history", source="historical_summaries", summarized_messages=2)
        ]


class TestSummaryPrompt:
    def test_attachments_are_named_in_the_transcript(self):
        image = ImageContent(base64_image="Zm9v", mime_type="image/png", meta={"file_path": "/tmp/shot.png"})
        pdf = FileContent(base64_data="Zm9v", mime_type="application/pdf", filename="q3.pdf")
        messages = [
            ChatMessage.from_user(content_parts=["review this", pdf]),
            tool_call("c1"),
            # An attachment a tool returned is nested inside the tool result rather than on the message.
            ChatMessage.from_tool(
                tool_result=[TextContent(text="captured"), image],
                origin=ToolCall(tool_name="browse", arguments={}, id="c1"),
            ),
        ]
        compactor = SummarizationCompactor(chat_generator=MockChatGenerator())
        prompt = compactor._prompt(messages=messages, indices=[0, 1, 2])
        transcript = prompt[1].text
        assert transcript is not None
        # The summary cannot reproduce either attachment, so the transcript has to name them well enough to ask again.
        assert "<file: q3.pdf, application/pdf>" in transcript
        assert "<image: image/png, file_path=/tmp/shot.png>" in transcript

    def test_custom_summary_instruction_replaces_the_default(self):
        compactor = SummarizationCompactor(
            chat_generator=MockChatGenerator(), summary_instruction="Only list file paths."
        )
        prompt = compactor._prompt(messages=[ChatMessage.from_user("task")], indices=[0])
        assert prompt[0].text == "Only list file paths."


class TestFailureHandling:
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
        compacted = SummarizationCompactor(generator, approximate_summary_tokens=1).compact(
            messages=messages, target_tokens=1, token_counter=COUNTER
        )
        assert compacted is not None
        assert len(prompts) == 2
        # The history was summarized before the step summary failed, and that progress is kept.
        assert summaries(messages=compacted) == [
            summary(text="history", source="historical_turns", summarized_messages=2)
        ]
        assert compacted[-2:] == messages[-2:]

    def test_raises_when_a_summary_does_not_shrink_the_conversation(self):
        messages = [
            ChatMessage.from_system("rules"),
            ChatMessage.from_user("old"),
            ChatMessage.from_assistant("answer"),
            ChatMessage.from_user("current"),
        ]
        compactor = SummarizationCompactor(
            MockChatGenerator("much longer summary " * 100), approximate_summary_tokens=1, raise_on_failure=True
        )
        with pytest.raises(RuntimeError, match="did not reduce"):
            compactor.compact(messages=messages, target_tokens=1, token_counter=COUNTER)

    @pytest.mark.parametrize(
        "generator_factory",
        [
            pytest.param(
                lambda: MockChatGenerator(response_fn=lambda messages: ChatMessage.from_assistant("")), id="empty-text"
            ),
            pytest.param(
                lambda: MockChatGenerator(
                    response_fn=lambda messages: ChatMessage.from_assistant(reasoning="I should summarize this.")
                ),
                id="reasoning-only",
            ),
            pytest.param(NoReplyGenerator, id="no-replies"),
        ],
    )
    def test_raises_when_the_generator_returns_no_usable_text(self, generator_factory):
        compactor = SummarizationCompactor(generator_factory(), raise_on_failure=True)
        with pytest.raises(RuntimeError, match="no usable text"):
            compactor.compact(messages=fresh_conversation_with_two_steps(), target_tokens=1, token_counter=COUNTER)


class TestConfiguration:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"min_keep_steps": -1}, "`min_keep_steps` must be at least 0"),
            ({"approximate_summary_tokens": 0}, "`approximate_summary_tokens` must be a positive"),
        ],
    )
    def test_rejects_invalid_settings(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            SummarizationCompactor(MockChatGenerator("summary"), **kwargs)

    def test_serde_round_trip(self):
        compactor = SummarizationCompactor(
            MockChatGenerator("summary"),
            min_keep_steps=2,
            approximate_summary_tokens=321,
            summary_instruction="custom",
            raise_on_failure=True,
        )
        restored = SummarizationCompactor.from_dict(compactor.to_dict())
        assert isinstance(restored.chat_generator, MockChatGenerator)
        assert restored.min_keep_steps == 2
        assert restored.approximate_summary_tokens == 321
        assert restored.summary_instruction == "custom"
        assert restored.raise_on_failure is True


class TestSummarizationCompactorInAgent:
    def test_compacts_history_through_a_compaction_hook(self):
        summary_generator = MockChatGenerator("summary")
        hook = CompactionHook(
            compactor=SummarizationCompactor(summary_generator, approximate_summary_tokens=64),
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
        assert summaries(messages=compacted) == [
            summary(text="summary", source="historical_turns", summarized_messages=2)
        ]
        assert all("old question" not in (message.text or "") for message in compacted)


class TestSummarizationCompactorAsync:
    @pytest.mark.asyncio
    async def test_compact_async_matches_compact(self):
        messages = fresh_conversation_with_two_steps()
        generator, prompts = summarizer("async summary")
        compacted = await SummarizationCompactor(generator, approximate_summary_tokens=1).compact_async(
            messages=messages, target_tokens=1, token_counter=COUNTER
        )
        assert len(prompts) == 1
        assert compacted == SummarizationCompactor(
            MockChatGenerator("async summary"), approximate_summary_tokens=1
        ).compact(messages=messages, target_tokens=1, token_counter=COUNTER)
