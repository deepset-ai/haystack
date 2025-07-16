# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack.components.builders.chat_prompt_builder import ChatPromptBuilder
from haystack.components.converters import OutputAdapter
from haystack.components.joiners import StringJoiner
from haystack.core.errors import BreakpointException
from haystack.core.pipeline.pipeline import Pipeline
from haystack.dataclasses import ChatMessage
from haystack.dataclasses.breakpoints import Breakpoint
from test.conftest import load_and_resume_pipeline_snapshot


class TestPipelineBreakpoints:
    @pytest.fixture
    def string_joiner_pipeline(self):
        pipeline = Pipeline()
        pipeline.add_component(
            "prompt_builder_1", ChatPromptBuilder(template=[ChatMessage.from_user("Builder 1: {{query}}")])
        )
        pipeline.add_component(
            "prompt_builder_2", ChatPromptBuilder(template=[ChatMessage.from_user("Builder 2: {{query}}")])
        )
        pipeline.add_component("adapter_1", OutputAdapter("{{messages[0].text}}", output_type=str))
        pipeline.add_component("adapter_2", OutputAdapter("{{messages[0].text}}", output_type=str))
        pipeline.add_component("string_joiner", StringJoiner())

        pipeline.connect("prompt_builder_1.prompt", "adapter_1.messages")
        pipeline.connect("prompt_builder_2.prompt", "adapter_2.messages")
        pipeline.connect("adapter_1", "string_joiner.strings")
        pipeline.connect("adapter_2", "string_joiner.strings")

        return pipeline

    @pytest.fixture(scope="session")
    def output_directory(self, tmp_path_factory):
        return tmp_path_factory.mktemp("output_files")

    BREAKPOINT_COMPONENTS = ["prompt_builder_1", "prompt_builder_2", "adapter_1", "adapter_2", "string_joiner"]

    @pytest.mark.parametrize("component", BREAKPOINT_COMPONENTS, ids=BREAKPOINT_COMPONENTS)
    @pytest.mark.integration
    def test_string_joiner_pipeline(self, string_joiner_pipeline, output_directory, component):
        string_1 = "What's Natural Language Processing?"
        string_2 = "What is life?"
        data = {"prompt_builder_1": {"query": string_1}, "prompt_builder_2": {"query": string_2}}

        # Create a Breakpoint on-the-fly using the shared output directory
        break_point = Breakpoint(component_name=component, visit_count=0, debug_path=str(output_directory))

        try:
            _ = string_joiner_pipeline.run(data, break_point=break_point)
        except BreakpointException:
            pass

        result = load_and_resume_pipeline_snapshot(
            pipeline=string_joiner_pipeline,
            output_directory=output_directory,
            component_name=break_point.component_name,
            data=data,
        )
        assert result["string_joiner"]
