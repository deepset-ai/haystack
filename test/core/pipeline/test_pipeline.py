# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import pytest

from haystack.components.joiners import BranchJoiner
from haystack.core.component import component
from haystack.core.component.types import InputSocket, OutputSocket, Variadic
from haystack.core.errors import PipelineRuntimeError
from haystack.core.pipeline import Pipeline


class TestPipeline:
    """
    This class contains only unit tests for the Pipeline class.
    It doesn't test Pipeline.run(), that is done separately in a different way.
    """

    def test__run_component_success(self):
        """Test successful component execution"""
        joiner_1 = BranchJoiner(type_=str)
        joiner_2 = BranchJoiner(type_=str)
        pp = Pipeline()
        pp.add_component("joiner_1", joiner_1)
        pp.add_component("joiner_2", joiner_2)
        pp.connect("joiner_1", "joiner_2")
        inputs = {"joiner_1": {"value": [{"sender": None, "value": "test_value"}]}}

        outputs = pp._run_component(
            component=pp._get_component_with_graph_metadata_and_visits("joiner_1", 0),
            inputs=inputs,
            component_visits={"joiner_1": 0, "joiner_2": 0},
        )

        assert outputs == {"value": "test_value"}
        # We remove input in greedy variadic sockets, even if they are from the user
        assert "value" not in inputs["joiner_1"]

    def test__run_component_fail(self):
        """Test error when component doesn't return a dictionary"""

        @component
        class WrongOutput:
            @component.output_types(output=str)
            def run(self, value: str):
                return "not_a_dict"

        wrong = WrongOutput()
        pp = Pipeline()
        pp.add_component("wrong", wrong)

        inputs = {"wrong": {"value": [{"sender": None, "value": "test_value"}]}}

        with pytest.raises(PipelineRuntimeError) as exc_info:
            pp._run_component(
                component=pp._get_component_with_graph_metadata_and_visits("wrong", 0),
                inputs=inputs,
                component_visits={"wrong": 0},
            )

        assert "didn't return a dictionary" in str(exc_info.value)
