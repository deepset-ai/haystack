# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0


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


def compact_each_round(
    compactor: SummarizationCompactor, messages: list[ChatMessage], rounds: list[list[ChatMessage]], target_tokens: int
) -> list[ChatMessage]:
    """
    Compact once per round, the way an Agent loop drives the hook as the conversation grows.

    Several behaviors only show up across compactions rather than within one, because a single `compact` call already
    takes enough of the conversation in one go to meet the target.
    """
    for addition in rounds:
        messages = [*messages, *addition]
        compacted = compactor.compact(messages=messages, target_tokens=target_tokens, token_counter=COUNTER)
        messages = compacted if compacted is not None else messages
    return messages


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


class TestTierOrder:
    """
    Which stretch of conversation is given up next.

    History is spent before the current task, and within each of the two, original messages are summarized before
    existing summaries are combined: `historical_turns`, `historical_summaries`, `current_task_steps`,
    `current_task_summaries`.
    """

    def test_summarizes_the_oldest_historical_turn_only(self):
        messages = two_turns_and_a_task()
        generator, prompts = summarizer("short historical summary")
        # Room for everything but the padded oldest turn, plus the summary standing in for it.
        target_tokens = COUNTER.count([messages[0], *messages[3:]]) + 100
        compacted = SummarizationCompactor(generator, approximate_summary_tokens=100).compact(
            messages=messages, target_tokens=target_tokens, token_counter=COUNTER
        )
        assert compacted is not None
        # Only the oldest turn was summarized, so the recent turn never reached the generator.
        assert len(prompts) == 1
        assert "oldest question" in prompts[0]
        assert "recent question" not in prompts[0]
        assert compacted == [messages[0], compacted[1], *messages[3:]]

    def test_summarizes_historical_turns_before_current_steps(self):
        messages = [
            ChatMessage.from_system("rules"),
            ChatMessage.from_user("old question " * 40),
            ChatMessage.from_assistant("old answer " * 40),
            *a_task_with_two_steps()[1:],
        ]
        generator, prompts = summarizer("history", "old step")
        compacted = SummarizationCompactor(generator, approximate_summary_tokens=1).compact(
            messages=messages, target_tokens=SMALLEST, token_counter=COUNTER
        )
        assert compacted is not None
        assert "old question" in prompts[0]
        assert "old result" in prompts[1]
        assert sources(compacted) == ["historical_turns", "current_task_steps"]
        # The newest step is never given up.
        assert compacted[-2:] == messages[-2:]

    def test_combines_historical_summaries_before_current_steps(self):
        messages = [
            ChatMessage.from_system("rules"),
            summary("first history " * 20, "historical_turns"),
            summary("second history " * 20, "historical_turns"),
            *a_task_with_two_steps()[1:],
        ]
        generator, prompts = summarizer("combined history", "old step")
        compacted = SummarizationCompactor(generator, approximate_summary_tokens=1).compact(
            messages=messages, target_tokens=SMALLEST, token_counter=COUNTER
        )
        assert compacted is not None
        # History holds no original messages, so its summaries are combined before any current-task step is touched.
        assert "first history" in prompts[0] and "old result" not in prompts[0]
        assert "old result" in prompts[1]
        assert sources(compacted) == ["historical_summaries", "current_task_steps"]

    def test_summarizes_steps_before_combining_current_task_summaries(self):
        messages = [
            ChatMessage.from_system("rules"),
            ChatMessage.from_user("current task"),
            summary("first step summary " * 20, "current_task_steps"),
            summary("second step summary " * 20, "current_task_steps"),
            *a_task_with_two_steps()[2:],
        ]
        generator, prompts = summarizer("old step", "combined steps")
        compacted = SummarizationCompactor(generator, approximate_summary_tokens=1).compact(
            messages=messages, target_tokens=SMALLEST, token_counter=COUNTER
        )
        assert compacted is not None
        # Summarizing the step frees more room than combining does, so it goes first. Only once no step may be
        # given up are the three summaries it left behind combined into one.
        assert "old result" in prompts[0]
        assert "first step summary" in prompts[1] and "old step" in prompts[1]
        assert sources(compacted) == ["current_task_summaries"]
        assert compacted[-2:] == messages[-2:]

    def test_combines_current_task_summaries_when_min_keep_steps_reserves_every_step(self):
        messages = [
            ChatMessage.from_system("rules"),
            ChatMessage.from_user("current task"),
            summary("first step summary " * 20, "current_task_steps"),
            summary("second step summary " * 20, "current_task_steps"),
            tool_call("new"),
            tool_result("new result", call_id="new"),
        ]
        generator, prompts = summarizer("combined steps")
        compacted = SummarizationCompactor(generator, min_keep_steps=1, approximate_summary_tokens=1).compact(
            messages=messages, target_tokens=SMALLEST, token_counter=COUNTER
        )
        assert compacted is not None
        # Combining spends no step, so `min_keep_steps` reserving the only one does not stand in its way.
        assert "first step summary" in prompts[0] and "second step summary" in prompts[0]
        assert sources(compacted) == ["current_task_summaries"]
        assert compacted[-2:] == messages[-2:]

    @pytest.mark.parametrize(("min_keep_steps", "expected"), [(0, 0), (1, 1), (2, 2), (20, 2)])
    def test_min_keep_steps_wins_over_an_unaffordable_target(self, min_keep_steps, expected):
        messages = a_task_with_two_steps()
        generator, _ = summarizer("step summary")
        compacted = SummarizationCompactor(
            generator, min_keep_steps=min_keep_steps, approximate_summary_tokens=1
        ).compact(messages=messages, target_tokens=SMALLEST, token_counter=COUNTER)
        result = compacted or messages
        assert sum(message.is_from(role=ChatRole.ASSISTANT) for message in result) == expected


class TestSummaryLifecycle:
    """
    How summaries build up and are combined as an Agent loop compacts the same conversation again and again.

    Combining is the last thing tried in each region, so summaries accumulate instead of being rewritten every time.
    """

    def test_historical_summaries_accumulate_while_raw_turns_remain(self):
        compactor = SummarizationCompactor(MockChatGenerator("summary"), approximate_summary_tokens=10)
        # Each round is another finished turn, and the newest user message anchors the current task.
        rounds = [
            [ChatMessage.from_user(f"question {index} " * 30), ChatMessage.from_assistant(f"answer {index} " * 30)]
            for index in range(4)
        ]
        # Loose enough that each round is paid for by summarizing one more turn, so combining is never reached.
        compacted = compact_each_round(compactor, [ChatMessage.from_system("rules")], rounds, target_tokens=1_100)
        # One summary per compaction rather than one combined summary, because turns were still there to give up.
        assert sources(compacted) == ["historical_turns", "historical_turns", "historical_turns"]

    def test_current_task_summaries_accumulate_while_raw_steps_remain(self):
        compactor = SummarizationCompactor(
            MockChatGenerator("summary"), min_keep_steps=1, approximate_summary_tokens=10
        )
        start = [ChatMessage.from_system("rules"), ChatMessage.from_user("current task")]
        rounds = [
            [tool_call(f"c{index}"), tool_result(f"result {index} " * 30, call_id=f"c{index}")] for index in range(4)
        ]
        compacted = compact_each_round(compactor, start, rounds, target_tokens=650)
        assert sources(compacted) == ["current_task_steps", "current_task_steps", "current_task_steps"]

    def test_stops_once_each_region_is_down_to_a_single_summary(self):
        # The floor compaction never goes below: the system block, one summary per region, the latest user message,
        # and the `min_keep_steps` newest steps.
        messages = [
            ChatMessage.from_system("rules"),
            summary("all history " * 20, "historical_summaries"),
            ChatMessage.from_user("current task"),
            summary("all earlier steps " * 20, "current_task_summaries"),
            tool_call("new"),
            tool_result("new result " * 20, call_id="new"),
        ]
        # No responses are queued, so any attempt to summarize would raise rather than quietly succeed.
        generator, prompts = summarizer()
        compacted = SummarizationCompactor(generator, min_keep_steps=1, approximate_summary_tokens=1).compact(
            messages=messages, target_tokens=SMALLEST, token_counter=COUNTER
        )
        assert compacted is None
        assert prompts == []

    def test_a_summarized_past_task_is_combined_into_history_once_a_new_task_arrives(self):
        # The summary of a task's early steps stops being a current-task summary the moment a newer user message
        # arrives, because the region a summary belongs to is decided by position, not by the `source` it records.
        messages = [
            ChatMessage.from_system("rules"),
            ChatMessage.from_user("past task " * 30),
            summary("early steps of the past task", "current_task_steps"),
            ChatMessage.from_assistant("late step " * 30),
            ChatMessage.from_user("current task"),
        ]
        generator, prompts = summarizer("past turn", "combined history")
        compacted = SummarizationCompactor(generator, approximate_summary_tokens=5).compact(
            messages=messages, target_tokens=SMALLEST, token_counter=COUNTER
        )
        assert compacted is not None
        # The rest of the past turn is summarized first, then that summary and the older one are combined.
        assert "past task" in prompts[0] and "early steps of the past task" not in prompts[0]
        assert "early steps of the past task" in prompts[1] and "past turn" in prompts[1]
        assert sources(compacted) == ["historical_summaries"]

    def test_summarized_messages_counts_what_a_summary_replaced(self):
        messages = [
            ChatMessage.from_system("rules"),
            ChatMessage.from_user("old " * 100),
            ChatMessage.from_assistant("answer " * 100),
            ChatMessage.from_user("task"),
        ]
        compacted = SummarizationCompactor(MockChatGenerator("summary"), approximate_summary_tokens=1).compact(
            messages=messages, target_tokens=SMALLEST, token_counter=COUNTER
        )
        assert compacted is not None
        assert compacted[1].is_from(role=ChatRole.USER)
        assert compacted[1].meta[_COMPACTION_META_KEY] == {
            "strategy": "summarization",
            "summarized_messages": 2,
            "source": "historical_turns",
        }

    def test_summarized_messages_counts_summaries_rather_than_the_messages_behind_them(self):
        messages = [
            ChatMessage.from_system("rules"),
            summary("first history " * 20, "historical_turns"),
            summary("second history " * 20, "historical_turns"),
            summary("third history " * 20, "historical_turns"),
            ChatMessage.from_user("current task"),
        ]
        compacted = SummarizationCompactor(MockChatGenerator("all history"), approximate_summary_tokens=1).compact(
            messages=messages, target_tokens=SMALLEST, token_counter=COUNTER
        )
        assert compacted is not None
        # Combining replaces summaries, so the count is three, not the many real messages those three stood for.
        assert compacted[1].meta[_COMPACTION_META_KEY]["summarized_messages"] == 3

    def test_leaves_the_input_conversation_untouched(self):
        messages = two_turns_and_a_task()
        SummarizationCompactor(MockChatGenerator("summary"), approximate_summary_tokens=1).compact(
            messages=messages, target_tokens=SMALLEST, token_counter=COUNTER
        )
        assert messages == two_turns_and_a_task()

    def test_returns_none_when_the_conversation_fits(self):
        generator, prompts = summarizer()
        messages = [ChatMessage.from_system("rules"), ChatMessage.from_user("task")]
        compacted = SummarizationCompactor(generator).compact(
            messages=messages, target_tokens=10_000, token_counter=COUNTER
        )
        assert compacted is None
        assert prompts == []


class TestSummaryContent:
    """What the summarizing Chat Generator is asked for, and under what budget."""

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
        SummarizationCompactor(generator, approximate_summary_tokens=1).compact(
            messages=messages, target_tokens=SMALLEST, token_counter=COUNTER
        )
        # The summary cannot reproduce either attachment, so the transcript has to name them well enough to ask again.
        assert "<file: q3.pdf, application/pdf>" in prompts[0]
        assert "<image: image/png, file_path=/tmp/shot.png>" in prompts[0]

    def test_custom_summary_instruction_replaces_the_default(self):
        generator, prompts = summarizer("summary")
        SummarizationCompactor(
            generator, summary_instruction="Only list file paths.", approximate_summary_tokens=1
        ).compact(messages=two_turns_and_a_task(), target_tokens=SMALLEST, token_counter=COUNTER)
        assert "Only list file paths." in prompts[0]
        assert "You are compacting one portion of a conversation" not in prompts[0]

    def test_the_instruction_reaches_the_model_verbatim(self):
        generator, prompts = summarizer("summary")
        SummarizationCompactor(
            generator, summary_instruction="Only list file paths.", approximate_summary_tokens=64
        ).compact(messages=two_turns_and_a_task(), target_tokens=SMALLEST, token_counter=COUNTER)
        # Nothing is appended, so what the model is told is exactly what the caller wrote.
        # `approximate_summary_tokens` is a planning estimate and never reaches the model.
        system_prompt = prompts[0].split("\n<conversation_to_summarize>")[0]
        assert system_prompt == "Only list file paths."
        assert "64" not in system_prompt


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
            MockChatGenerator("much longer summary " * 100), approximate_summary_tokens=1, raise_on_failure=True
        )
        with pytest.raises(RuntimeError, match="did not reduce"):
            compactor.compact(messages=messages, target_tokens=SMALLEST, token_counter=COUNTER)

    @pytest.mark.parametrize(
        "generator_factory",
        [
            pytest.param(
                lambda: MockChatGenerator(response_fn=lambda messages: ChatMessage.from_assistant("")), id="empty-text"
            ),
            pytest.param(
                lambda: MockChatGenerator(response_fn=lambda messages: ChatMessage.from_assistant("  \n  ")),
                id="whitespace-only-text",
            ),
            pytest.param(NoReplyGenerator, id="no-replies"),
        ],
    )
    def test_raises_when_the_generator_returns_no_usable_text(self, generator_factory):
        compactor = SummarizationCompactor(generator_factory(), raise_on_failure=True)
        with pytest.raises(RuntimeError, match="no usable text"):
            compactor.compact(messages=two_turns_and_a_task(), target_tokens=SMALLEST, token_counter=COUNTER)

    def test_an_unusable_reply_is_reported_with_the_generator_output(self):
        # The reply is discarded once compaction moves on, so the error carries it: `finish_reason` is usually what
        # says why the summary came back unusable.
        truncated = ChatMessage.from_assistant("", meta={"finish_reason": "length"})
        compactor = SummarizationCompactor(
            MockChatGenerator(response_fn=lambda messages: truncated), raise_on_failure=True
        )
        with pytest.raises(RuntimeError) as failure:
            compactor.compact(messages=two_turns_and_a_task(), target_tokens=SMALLEST, token_counter=COUNTER)
        assert "'finish_reason': 'length'" in str(failure.value)


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
        assert sources(compacted) == ["historical_turns"]
        assert all("old question" not in (message.text or "") for message in compacted)


class TestSummarizationCompactorAsync:
    @pytest.mark.asyncio
    async def test_compact_async_matches_compact(self):
        messages = two_turns_and_a_task()
        generator, prompts = summarizer("async summary")
        compacted = await SummarizationCompactor(generator, approximate_summary_tokens=1).compact_async(
            messages=messages, target_tokens=SMALLEST, token_counter=COUNTER
        )
        assert len(prompts) == 1
        assert compacted == SummarizationCompactor(
            MockChatGenerator("async summary"), approximate_summary_tokens=1
        ).compact(messages=messages, target_tokens=SMALLEST, token_counter=COUNTER)
