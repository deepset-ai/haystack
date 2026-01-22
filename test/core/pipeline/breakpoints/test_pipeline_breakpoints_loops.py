# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from haystack import component
from haystack.components.builders import ChatPromptBuilder
from haystack.core.errors import BreakpointException
from haystack.core.pipeline.pipeline import Pipeline
from haystack.dataclasses import ChatMessage
from haystack.dataclasses.breakpoints import Breakpoint


@component
class OutputValidator:
    def __init__(self, pydantic_model: Any):
        self.pydantic_model = pydantic_model
        self.iteration_counter = 0

    @component.output_types(valid_replies=list[ChatMessage], invalid_replies=list[ChatMessage], error_message=str)
    def run(self, replies: list[ChatMessage]) -> dict[str, list[ChatMessage] | str]:
        self.iteration_counter += 1
        try:
            assert replies[0].text is not None
            output_dict = json.loads(replies[0].text)
            self.pydantic_model.model_validate(output_dict)
            return {"valid_replies": replies}
        except (ValueError, ValidationError) as e:
            return {"invalid_replies": replies, "error_message": str(e)}


@component
class FakeChatGenerator:
    def __init__(self, response: str):
        self.response = response

    @component.output_types(replies=list[ChatMessage])
    def run(self, messages: list[ChatMessage]) -> dict[str, list[ChatMessage]]:
        return {"replies": [ChatMessage.from_assistant(self.response)]}


class City(BaseModel):
    name: str
    country: str
    population: int


class CitiesData(BaseModel):
    cities: list[City]


class TestPipelineBreakpointsLoops:
    """
    This class contains tests for pipelines with validation loops and breakpoints.
    """

    @pytest.fixture
    def validation_loop_pipeline(self):
        """Create a pipeline with validation loops for testing."""
        prompt_template = [
            ChatMessage.from_user(
                """
                Create a JSON object from the information present in this passage: {{passage}}.
                Only use information that is present in the passage. Follow this JSON schema, but only return the
                 actual instances without any additional schema definition:
                {{schema}}
                Make sure your response is a dict and not a list.
                {% if invalid_replies and error_message %}
                  You already created the following output in a previous attempt: {{invalid_replies}}
                  However, this doesn't comply with the format requirements from above and triggered this
                   Python exception: {{error_message}}
                  Correct the output and try again. Just return the corrected output without any extra explanations.
                {% endif %}
                """
            )
        ]

        response_json = json.dumps(
            {
                "cities": [
                    {"name": "Berlin", "country": "Germany", "population": 3850809},
                    {"name": "Paris", "country": "France", "population": 2161000},
                    {"name": "Lisbon", "country": "Portugal", "population": 504718},
                ]
            }
        )

        pipeline = Pipeline(max_runs_per_component=5)
        pipeline.add_component(instance=ChatPromptBuilder(template=prompt_template), name="prompt_builder")
        pipeline.add_component(instance=FakeChatGenerator(response=response_json), name="llm")
        pipeline.add_component(instance=OutputValidator(pydantic_model=CitiesData), name="output_validator")

        pipeline.connect("prompt_builder.prompt", "llm.messages")
        pipeline.connect("llm.replies", "output_validator")
        pipeline.connect("output_validator.invalid_replies", "prompt_builder.invalid_replies")
        pipeline.connect("output_validator.error_message", "prompt_builder.error_message")

        return pipeline

    BREAKPOINT_COMPONENTS = ["prompt_builder", "llm", "output_validator"]

    @pytest.mark.parametrize("component", BREAKPOINT_COMPONENTS, ids=BREAKPOINT_COMPONENTS)
    @pytest.mark.integration
    def test_pipeline_breakpoints_validation_loop(
        self, validation_loop_pipeline, output_directory, component, load_and_resume_pipeline_snapshot
    ):
        """
        Test that a pipeline with validation loops can be executed with breakpoints at each component.
        """
        data = {"prompt_builder": {"passage": "Berlin, Paris, Lisbon...", "schema": "CitiesData schema"}}

        # Create a Breakpoint on-the-fly using the shared output directory
        break_point = Breakpoint(component_name=component, visit_count=0, snapshot_file_path=str(output_directory))

        try:
            _ = validation_loop_pipeline.run(data, break_point=break_point)
        except BreakpointException:
            pass

        result = load_and_resume_pipeline_snapshot(
            pipeline=validation_loop_pipeline,
            output_directory=output_directory,
            component_name=break_point.component_name,
            data=data,
        )

        assert "output_validator" in result
        assert "valid_replies" in result["output_validator"]
        valid_reply = result["output_validator"]["valid_replies"][0].text
        valid_json = json.loads(valid_reply)
        assert "cities" in valid_json
        assert len(valid_json["cities"]) == 3
        cities_data = CitiesData.model_validate(valid_json)
        assert len(cities_data.cities) == 3
        assert cities_data.cities[0].name == "Berlin"
        assert cities_data.cities[1].name == "Paris"
        assert cities_data.cities[2].name == "Lisbon"
