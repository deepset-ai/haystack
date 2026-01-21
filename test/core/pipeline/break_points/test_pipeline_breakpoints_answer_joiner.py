# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack import component
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.joiners import AnswerJoiner
from haystack.core.errors import BreakpointException
from haystack.core.pipeline.pipeline import Pipeline
from haystack.dataclasses import ChatMessage
from haystack.dataclasses.breakpoints import Breakpoint


@component
class FakeChatGenerator:
    def __init__(self, content: str, model_name: str):
        self.content = content
        self.model_name = model_name

    @component.output_types(replies=list[ChatMessage])
    def run(self, messages: list[ChatMessage]):
        return {"replies": [ChatMessage.from_assistant(self.content)]}


class TestPipelineBreakpoints:
    @pytest.fixture
    def answer_join_pipeline(self):
        """Creates a pipeline with fake components."""
        pipeline = Pipeline()
        pipeline.add_component("gpt-4o", FakeChatGenerator("GPT-4 response", "gpt-4o"))
        pipeline.add_component("gpt-3", FakeChatGenerator("GPT-3 response", "gpt-3.5-turbo"))
        pipeline.add_component("answer_builder_a", AnswerBuilder())
        pipeline.add_component("answer_builder_b", AnswerBuilder())
        pipeline.add_component("answer_joiner", AnswerJoiner())
        pipeline.connect("gpt-4o.replies", "answer_builder_a")
        pipeline.connect("gpt-3.replies", "answer_builder_b")
        pipeline.connect("answer_builder_a.answers", "answer_joiner")
        pipeline.connect("answer_builder_b.answers", "answer_joiner")

        return pipeline

    BREAKPOINT_COMPONENTS = ["gpt-4o", "gpt-3", "answer_builder_a", "answer_builder_b", "answer_joiner"]

    @pytest.mark.parametrize("component", BREAKPOINT_COMPONENTS, ids=BREAKPOINT_COMPONENTS)
    @pytest.mark.integration
    def test_pipeline_breakpoints_answer_joiner(
        self, answer_join_pipeline, output_directory, component, load_and_resume_pipeline_snapshot
    ):
        """
        Test that an answer joiner pipeline can be executed with breakpoints at each component.
        """
        query = "What's Natural Language Processing?"
        messages = [
            ChatMessage.from_system("You are a helpful, respectful and honest assistant. Be super concise."),
            ChatMessage.from_user(query),
        ]
        data = {
            "gpt-4o": {"messages": messages},
            "gpt-3": {"messages": messages},
            "answer_builder_a": {"query": query},
            "answer_builder_b": {"query": query},
        }

        # Create a Breakpoint on-the-fly using the shared output directory
        break_point = Breakpoint(component_name=component, visit_count=0, snapshot_file_path=str(output_directory))

        try:
            _ = answer_join_pipeline.run(data, break_point=break_point)
        except BreakpointException:
            pass

        result = load_and_resume_pipeline_snapshot(
            pipeline=answer_join_pipeline,
            output_directory=output_directory,
            component_name=break_point.component_name,
            data=data,
        )
        assert result["answer_joiner"]
        assert len(result["answer_joiner"]["answers"]) == 2
        assert "GPT-4 response" in [a.data for a in result["answer_joiner"]["answers"]]
        assert "GPT-3 response" in [a.data for a in result["answer_joiner"]["answers"]]
