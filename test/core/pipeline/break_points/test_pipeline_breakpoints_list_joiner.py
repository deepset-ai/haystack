# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack import Pipeline, component
from haystack.components.builders import ChatPromptBuilder
from haystack.components.joiners import ListJoiner
from haystack.core.errors import BreakpointException
from haystack.dataclasses import ChatMessage
from haystack.dataclasses.breakpoints import Breakpoint


@component
class FakeChatGenerator:
    def __init__(self, response: str):
        self.response = response

    @component.output_types(replies=list[ChatMessage])
    def run(self, messages: list[ChatMessage], **kwargs):
        return {"replies": [ChatMessage.from_assistant(self.response)]}


class TestPipelineBreakpoints:
    @pytest.fixture
    def list_joiner_pipeline(self):
        user_message = [ChatMessage.from_user("Give a brief answer the following question: {{query}}")]

        feedback_prompt = """
            You are given a question and an answer.
            Your task is to provide a score and a brief feedback on the answer.
            Question: {{query}}
            Answer: {{response}}
            """
        feedback_message = [ChatMessage.from_system(feedback_prompt)]

        pipe = Pipeline()
        pipe.add_component("prompt_builder", ChatPromptBuilder(template=user_message))
        pipe.add_component("llm", FakeChatGenerator("Nuclear physics is the study of atomic nuclei."))
        pipe.add_component("feedback_prompt_builder", ChatPromptBuilder(template=feedback_message))
        pipe.add_component("feedback_llm", FakeChatGenerator("Score: 8/10. Concise and accurate."))
        pipe.add_component("list_joiner", ListJoiner(list[ChatMessage]))

        pipe.connect("prompt_builder.prompt", "llm.messages")
        pipe.connect("prompt_builder.prompt", "list_joiner")
        pipe.connect("llm.replies", "list_joiner")
        pipe.connect("llm.replies", "feedback_prompt_builder.response")
        pipe.connect("feedback_prompt_builder.prompt", "feedback_llm.messages")
        pipe.connect("feedback_llm.replies", "list_joiner")

        return pipe

    BREAKPOINT_COMPONENTS = ["prompt_builder", "llm", "feedback_prompt_builder", "feedback_llm", "list_joiner"]

    @pytest.mark.parametrize("component", BREAKPOINT_COMPONENTS, ids=BREAKPOINT_COMPONENTS)
    @pytest.mark.integration
    def test_list_joiner_pipeline(
        self, list_joiner_pipeline, output_directory, component, load_and_resume_pipeline_snapshot
    ):
        query = "What is nuclear physics?"
        data = {
            "prompt_builder": {"template_variables": {"query": query}},
            "feedback_prompt_builder": {"template_variables": {"query": query}},
        }

        # Create a Breakpoint on-the-fly using the shared output directory
        break_point = Breakpoint(component_name=component, visit_count=0, snapshot_file_path=str(output_directory))

        try:
            _ = list_joiner_pipeline.run(data, break_point=break_point)
        except BreakpointException:
            pass

        result = load_and_resume_pipeline_snapshot(
            pipeline=list_joiner_pipeline,
            output_directory=output_directory,
            component_name=break_point.component_name,
            data=data,
        )
        assert result["list_joiner"]
        messages = result["list_joiner"]["values"]
        assert len(messages) == 3
        assert any("Nuclear physics is the study of atomic nuclei." in str(m) for m in messages)
        assert any("Score: 8/10." in str(m) for m in messages)
