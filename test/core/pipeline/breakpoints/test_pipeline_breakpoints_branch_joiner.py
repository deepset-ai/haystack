# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any

import pytest

from haystack import component
from haystack.components.converters import OutputAdapter
from haystack.components.joiners import BranchJoiner
from haystack.components.validators import JsonSchemaValidator
from haystack.core.errors import BreakpointException
from haystack.core.pipeline.pipeline import Pipeline
from haystack.dataclasses import ChatMessage
from haystack.dataclasses.breakpoints import Breakpoint


@component
class FakeChatGenerator:
    def __init__(self, content: str):
        self.content = content

    @component.output_types(replies=list[ChatMessage])
    def run(
        self, messages: list[ChatMessage], generation_kwargs: dict | None = None, **kwargs: Any
    ) -> dict[str, list[ChatMessage]]:
        return {"replies": [ChatMessage.from_assistant(self.content)]}


class TestPipelineBreakpoints:
    @pytest.fixture
    def branch_joiner_pipeline(self):
        person_schema = {
            "type": "object",
            "properties": {
                "first_name": {"type": "string", "pattern": "^[A-Z][a-z]+$"},
                "last_name": {"type": "string", "pattern": "^[A-Z][a-z]+$"},
                "nationality": {"type": "string", "enum": ["Italian", "Portuguese", "American"]},
            },
            "required": ["first_name", "last_name", "nationality"],
        }

        content = '{"first_name": "Peter", "last_name": "Parker", "nationality": "American"}'

        pipe = Pipeline()
        pipe.add_component("joiner", BranchJoiner(list[ChatMessage]))
        pipe.add_component("fc_llm", FakeChatGenerator(content))
        pipe.add_component("validator", JsonSchemaValidator(json_schema=person_schema))
        pipe.add_component("adapter", OutputAdapter("{{chat_message}}", list[ChatMessage], unsafe=True))

        pipe.connect("adapter", "joiner")
        pipe.connect("joiner", "fc_llm")
        pipe.connect("fc_llm.replies", "validator.messages")
        pipe.connect("validator.validation_error", "joiner")

        return pipe

    BREAKPOINT_COMPONENTS = ["joiner", "fc_llm", "validator", "adapter"]

    @pytest.mark.parametrize("component", BREAKPOINT_COMPONENTS, ids=BREAKPOINT_COMPONENTS)
    @pytest.mark.integration
    def test_pipeline_breakpoints_branch_joiner(
        self, branch_joiner_pipeline, output_directory, component, load_and_resume_pipeline_snapshot
    ):
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
        assert result["validator"]
        valid_json = json.loads(result["validator"]["validated"][0].text)
        assert valid_json["first_name"] == "Peter"
        assert valid_json["last_name"] == "Parker"
        assert valid_json["nationality"] == "American"
