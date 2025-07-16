# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.joiners import ListJoiner
from haystack.core.errors import BreakpointException
from haystack.dataclasses import ChatMessage
from haystack.dataclasses.breakpoints import Breakpoint
from haystack.utils.auth import Secret
from test.conftest import load_and_resume_pipeline_snapshot


class TestPipelineBreakpoints:
    @pytest.fixture
    def mock_openai_chat_generator(self):
        """
        Creates a mock for the OpenAIChatGenerator.
        """
        with patch("openai.resources.chat.completions.Completions.create") as mock_chat_completion_create:
            # Create mock completion objects
            mock_completion = MagicMock()
            mock_completion.choices = [
                MagicMock(
                    finish_reason="stop",
                    index=0,
                    message=MagicMock(
                        content="Nuclear physics is the study of atomic nuclei, their constituents, "
                        "and their interactions."
                    ),
                )
            ]
            mock_completion.usage = {"prompt_tokens": 57, "completion_tokens": 40, "total_tokens": 97}

            mock_chat_completion_create.return_value = mock_completion

            # Create a mock for the OpenAIChatGenerator
            @patch.dict("os.environ", {"OPENAI_API_KEY": "test-api-key"})
            def create_mock_generator(model_name):
                generator = OpenAIChatGenerator(model=model_name, api_key=Secret.from_env_var("OPENAI_API_KEY"))

                # Mock the run method
                def mock_run(messages, streaming_callback=None, generation_kwargs=None, tools=None, tools_strict=None):
                    # Check if this is a feedback request or a regular query
                    if any("feedback" in msg.text.lower() for msg in messages):
                        content = (
                            "Score: 8/10. The answer is concise and accurate, providing a good overview "
                            "of nuclear physics."
                        )
                    else:
                        content = (
                            "Nuclear physics is the study of atomic nuclei, their constituents, and their interactions."
                        )

                    return {
                        "replies": [ChatMessage.from_assistant(content)],
                        "meta": {
                            "model": model_name,
                            "usage": {"prompt_tokens": 57, "completion_tokens": 40, "total_tokens": 97},
                        },
                    }

                # Replace the run method with our mock
                generator.run = mock_run

                return generator

            yield create_mock_generator

    @pytest.fixture
    def list_joiner_pipeline(self, mock_openai_chat_generator):
        user_message = [ChatMessage.from_user("Give a brief answer the following question: {{query}}")]

        feedback_prompt = """
            You are given a question and an answer.
            Your task is to provide a score and a brief feedback on the answer.
            Question: {{query}}
            Answer: {{response}}
            """

        feedback_message = [ChatMessage.from_system(feedback_prompt)]

        prompt_builder = ChatPromptBuilder(template=user_message)
        feedback_prompt_builder = ChatPromptBuilder(template=feedback_message)
        llm = mock_openai_chat_generator("gpt-4o-mini")
        feedback_llm = mock_openai_chat_generator("gpt-4o-mini")

        pipe = Pipeline()
        pipe.add_component("prompt_builder", prompt_builder)
        pipe.add_component("llm", llm)
        pipe.add_component("feedback_prompt_builder", feedback_prompt_builder)
        pipe.add_component("feedback_llm", feedback_llm)
        pipe.add_component("list_joiner", ListJoiner(List[ChatMessage]))

        pipe.connect("prompt_builder.prompt", "llm.messages")
        pipe.connect("prompt_builder.prompt", "list_joiner")
        pipe.connect("llm.replies", "list_joiner")
        pipe.connect("llm.replies", "feedback_prompt_builder.response")
        pipe.connect("feedback_prompt_builder.prompt", "feedback_llm.messages")
        pipe.connect("feedback_llm.replies", "list_joiner")

        return pipe

    @pytest.fixture(scope="session")
    def output_directory(self, tmp_path_factory) -> Path:
        return tmp_path_factory.mktemp("output_files")

    BREAKPOINT_COMPONENTS = ["prompt_builder", "llm", "feedback_prompt_builder", "feedback_llm", "list_joiner"]

    @pytest.mark.parametrize("component", BREAKPOINT_COMPONENTS, ids=BREAKPOINT_COMPONENTS)
    @pytest.mark.integration
    def test_list_joiner_pipeline(self, list_joiner_pipeline, output_directory, component):
        query = "What is nuclear physics?"
        data = {
            "prompt_builder": {"template_variables": {"query": query}},
            "feedback_prompt_builder": {"template_variables": {"query": query}},
        }

        # Create a Breakpoint on-the-fly using the shared output directory
        break_point = Breakpoint(component_name=component, visit_count=0, debug_path=str(output_directory))

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
