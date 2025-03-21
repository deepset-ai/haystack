# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor

import pytest

from haystack.components.joiners import BranchJoiner
from haystack.core.component import component, Component
from haystack.core.errors import PipelineRuntimeError
from haystack.core.pipeline import Pipeline


class TestPipeline:
    """
    This class contains only unit tests for the Pipeline class.
    It doesn't test Pipeline.run(), that is done separately in a different way.
    """

    def test_pipeline_thread_safety(self, waiting_component, spying_tracer):
        # Initialize pipeline with synchronous components
        pp = Pipeline()
        pp.add_component("wait", waiting_component())

        run_data = [{"wait_for": 1}, {"wait_for": 2}]

        # Use ThreadPoolExecutor to run pipeline calls in parallel
        with ThreadPoolExecutor(max_workers=len(run_data)) as executor:
            # Submit pipeline runs to the executor
            futures = [executor.submit(pp.run, data) for data in run_data]

            # Wait for all futures to complete
            for future in futures:
                future.result()

        # Verify component visits using tracer
        component_spans = [sp for sp in spying_tracer.spans if sp.operation_name == "haystack.component.run"]

        for span in component_spans:
            assert span.tags["haystack.component.visits"] == 1

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

    def test_run(self):
        joiner_1 = BranchJoiner(type_=str)
        joiner_2 = BranchJoiner(type_=str)
        pp = Pipeline()
        pp.add_component("joiner_1", joiner_1)
        pp.add_component("joiner_2", joiner_2)
        pp.connect("joiner_1", "joiner_2")

        _ = pp.run({"value": "test_value"})

    def test_validate_breakpoints(self):
        # simple pipeline
        joiner_1 = BranchJoiner(type_=str)
        joiner_2 = BranchJoiner(type_=str)
        pipeline = Pipeline()
        pipeline.add_component("comp1", joiner_1)
        pipeline.add_component("comp2", joiner_2)
        pipeline.connect("comp1", "comp2")

        # valid breakpoints
        breakpoints = {("comp1", 0), ("comp2", 1)}
        validated = pipeline._validate_breakpoints(breakpoints)
        assert validated == {("comp1", 0), ("comp2", 1)}

        # should default to 0
        breakpoints = {("comp1", None), ("comp2", 1)}
        validated = pipeline._validate_breakpoints(breakpoints)
        assert validated == {("comp1", 0), ("comp2", 1)}

        # should remain as it is
        breakpoints = {("comp1", -1)}
        validated = pipeline._validate_breakpoints(breakpoints)
        assert validated == {("comp1", -1)}

        # contains invalid components
        breakpoints = {("comp1", 0), ("non_existent_component", 1)}
        with pytest.raises(ValueError, match="Breakpoint .* is not a registered component"):
            pipeline._validate_breakpoints(breakpoints)

        # no breakpoints are defined
        breakpoints = set()
        validated = pipeline._validate_breakpoints(breakpoints)
        assert validated == set()
