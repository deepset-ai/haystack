# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from haystack.components.converters import OutputAdapter
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.joiners import BranchJoiner
from haystack.components.validators import JsonSchemaValidator
from haystack.core.errors import BreakpointException
from haystack.core.pipeline.pipeline import Pipeline
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
                        content='{"first_name": "Peter", "last_name": "Parker", "nationality": "American"}'
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
                    content = '{"first_name": "Peter", "last_name": "Parker", "nationality": "American"}'

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
    def branch_joiner_pipeline(self, mock_openai_chat_generator):
        person_schema = {
            "type": "object",
            "properties": {
                "first_name": {"type": "string", "pattern": "^[A-Z][a-z]+$"},
                "last_name": {"type": "string", "pattern": "^[A-Z][a-z]+$"},
                "nationality": {"type": "string", "enum": ["Italian", "Portuguese", "American"]},
            },
            "required": ["first_name", "last_name", "nationality"],
        }

        pipe = Pipeline()
        pipe.add_component("joiner", BranchJoiner(list[ChatMessage]))
        pipe.add_component("fc_llm", mock_openai_chat_generator("gpt-4o-mini"))
        pipe.add_component("validator", JsonSchemaValidator(json_schema=person_schema))
        pipe.add_component("adapter", OutputAdapter("{{chat_message}}", list[ChatMessage], unsafe=True))

        pipe.connect("adapter", "joiner")
        pipe.connect("joiner", "fc_llm")
        pipe.connect("fc_llm.replies", "validator.messages")
        pipe.connect("validator.validation_error", "joiner")

        return pipe

    @pytest.fixture(scope="session")
    def output_directory(self, tmp_path_factory) -> Path:
        return tmp_path_factory.mktemp("output_files")

    BREAKPOINT_COMPONENTS = ["joiner", "fc_llm", "validator", "adapter"]

    @pytest.mark.parametrize("component", BREAKPOINT_COMPONENTS, ids=BREAKPOINT_COMPONENTS)
    @pytest.mark.integration
    def test_pipeline_breakpoints_branch_joiner(self, branch_joiner_pipeline, output_directory, component):
        data = {
            "fc_llm": {"generation_kwargs": {"response_format": {"type": "json_object"}}},
            "adapter": {"chat_message": [ChatMessage.from_user("Create JSON from Peter Parker")]},
        }

        # Create a Breakpoint on-the-fly using the shared output directory
        break_point = Breakpoint(component_name=component, visit_count=0, snapshot_file_path=str(output_directory))

        try:
            _ = branch_joiner_pipeline.run(data, break_point=break_point)
        except BreakpointException:
            pass

        result = load_and_resume_pipeline_snapshot(
            pipeline=branch_joiner_pipeline,
            output_directory=output_directory,
            component_name=break_point.component_name,
            data=data,
        )
        assert result["validator"], "The result should be valid according to the schema."
