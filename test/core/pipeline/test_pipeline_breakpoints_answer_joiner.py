# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest

from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.joiners import AnswerJoiner
from haystack.core.errors import BreakpointException
from haystack.core.pipeline.pipeline import Pipeline
from haystack.dataclasses import ChatMessage
from haystack.dataclasses.breakpoints import Breakpoint
from haystack.utils.auth import Secret
from test.conftest import load_and_resume_pipeline_state


class TestPipelineBreakpoints:
    @pytest.fixture
    def mock_openai_chat_generator(self):
        with patch("openai.resources.chat.completions.Completions.create") as mock_chat_completion_create:
            mock_completion = MagicMock()
            mock_completion.choices = [
                MagicMock(
                    finish_reason="stop",
                    index=0,
                    message=MagicMock(
                        content="Natural Language Processing (NLP) is a field of AI focused on enabling "
                        "computers to understand, interpret, and generate human language."
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
                    if "gpt-4" in model_name:
                        content = (
                            "Natural Language Processing (NLP) is a field of AI focused on enabling computers "
                            "to understand, interpret, and generate human language."
                        )
                    else:
                        content = "NLP is a branch of AI that helps machines understand and process human language."

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
    def answer_join_pipeline(self, mock_openai_chat_generator):
        """
        Creates a pipeline with mocked OpenAI components.
        """
        # Create the pipeline with mocked components
        pipeline = Pipeline(connection_type_validation=False)
        pipeline.add_component("gpt-4o", mock_openai_chat_generator("gpt-4o"))
        pipeline.add_component("gpt-3", mock_openai_chat_generator("gpt-3.5-turbo"))
        pipeline.add_component("answer_builder_a", AnswerBuilder())
        pipeline.add_component("answer_builder_b", AnswerBuilder())
        pipeline.add_component("answer_joiner", AnswerJoiner())
        pipeline.connect("gpt-4o.replies", "answer_builder_a")
        pipeline.connect("gpt-3.replies", "answer_builder_b")
        pipeline.connect("answer_builder_a.answers", "answer_joiner")
        pipeline.connect("answer_builder_b.answers", "answer_joiner")

        return pipeline

    @pytest.fixture(scope="session")
    def output_directory(self, tmp_path_factory):
        return tmp_path_factory.mktemp("output_files")

    components = [
        Breakpoint("gpt-4o", 0),
        Breakpoint("gpt-3", 0),
        Breakpoint("answer_builder_a", 0),
        Breakpoint("answer_builder_b", 0),
        Breakpoint("answer_joiner", 0),
    ]

    @pytest.mark.parametrize("component", components)
    @pytest.mark.integration
    def test_pipeline_breakpoints_answer_joiner(self, answer_join_pipeline, output_directory, component):
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

        try:
            _ = answer_join_pipeline.run(data, break_point=component, debug_path=str(output_directory))
        except BreakpointException:
            pass

        result = load_and_resume_pipeline_state(answer_join_pipeline, output_directory, component.component_name, data)
        assert result["answer_joiner"]
