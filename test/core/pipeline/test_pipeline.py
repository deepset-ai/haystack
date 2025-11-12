# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor

import pytest

from haystack.components.joiners import BranchJoiner
from haystack.core.component import component
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

        run_data = [{"wait_for": 0.001}, {"wait_for": 0.002}]

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

    def test_prepare_component_inputs(self):
        joiner_1 = BranchJoiner(type_=str)
        joiner_2 = BranchJoiner(type_=str)
        pp = Pipeline()
        component_name = "joiner_1"
        pp.add_component(component_name, joiner_1)
        pp.add_component("joiner_2", joiner_2)
        pp.connect(component_name, "joiner_2")
        inputs = {"joiner_1": {"value": [{"sender": None, "value": "test_value"}]}}
        comp_dict = pp._get_component_with_graph_metadata_and_visits(component_name, 0)

        _ = pp._consume_component_inputs(component_name=component_name, component=comp_dict, inputs=inputs)
        # We remove input in greedy variadic sockets, even if they are from the user
        assert inputs == {"joiner_1": {}}

    def test__run_component_success(self):
        """Test successful component execution"""
        joiner_1 = BranchJoiner(type_=str)
        joiner_2 = BranchJoiner(type_=str)
        pp = Pipeline()
        component_name = "joiner_1"
        pp.add_component(component_name, joiner_1)
        pp.add_component("joiner_2", joiner_2)
        pp.connect(component_name, "joiner_2")
        inputs = {"value": ["test_value"]}

        outputs = pp._run_component(
            component_name=component_name,
            component=pp._get_component_with_graph_metadata_and_visits(component_name, 0),
            inputs=inputs,
            component_visits={component_name: 0, "joiner_2": 0},
        )

        assert outputs == {"value": "test_value"}

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
        inputs = {"value": "test_value"}

        with pytest.raises(PipelineRuntimeError) as exc_info:
            pp._run_component(
                component_name="wrong",
                component=pp._get_component_with_graph_metadata_and_visits("wrong", 0),
                inputs=inputs,
                component_visits={"wrong": 0},
            )

        assert "Expected a dict" in str(exc_info.value)

    def test_run_component_error(self):
        """Test error when component fails to run"""

        @component
        class ErroringComponent:
            @component.output_types(output=str)
            def run(self):
                raise ValueError("Test error")

        erroring_component = ErroringComponent()
        pp = Pipeline()
        pp.add_component("erroring_component", erroring_component)

        inputs = {"wrong": {"value": [{"sender": None, "value": "test_value"}]}}

        with pytest.raises(PipelineRuntimeError) as exc_info:
            pp._run_component(
                component_name="erroring_component",
                component=pp._get_component_with_graph_metadata_and_visits("erroring_component", 0),
                inputs=inputs,
                component_visits={"erroring_component": 0},
            )
        assert "Component name: 'erroring_component'" in str(exc_info.value)

    def test_component_with_all_outputs_consumed_appears_in_results(self):
        """Test that components with all outputs consumed by downstream components appear in results with empty dict"""
        from typing import Optional

        @component
        class Producer:
            def __init__(self, prefix: str):
                self.prefix = prefix

            @component.output_types(value=Optional[str])
            def run(self, text: Optional[str]):
                return {"value": f"{self.prefix}: {text}"}

        @component
        class EmptyProcessor:
            @component.output_types()
            def run(self, sources: list[str]):
                # Returns empty dict when sources is empty
                return {}

        @component
        class Combiner:
            @component.output_types(combined=str)
            def run(self, input_a: Optional[str], input_b: Optional[str]):
                if input_a is None:
                    input_a = ""
                if input_b is None:
                    input_b = ""
                return {"combined": f"{input_a} | {input_b}"}

        pp = Pipeline()
        pp.add_component("producer_a", Producer("A"))
        pp.add_component("producer_b", Producer("B"))
        pp.add_component("empty_processor", EmptyProcessor())
        pp.add_component("combiner", Combiner())

        pp.connect("producer_a.value", "combiner.input_a")
        pp.connect("producer_b.value", "combiner.input_b")

        result = pp.run(
            {"producer_a": {"text": "hello"}, "producer_b": {"text": "world"}, "empty_processor": {"sources": []}},
            include_outputs_from={"producer_a", "empty_processor", "combiner"},
        )

        # Producer A should appear in results because it's in include_outputs_from
        assert "producer_a" in result
        assert result["producer_a"] == {"value": "A: hello"}
        # Producer B should NOT appear since it's not in include_outputs_from
        assert "producer_b" not in result
        # Combiner should appear in results
        assert "combiner" in result
        assert result["combiner"] == {"combined": "A: hello | B: world"}
        # Empty processor should appear in results even though it returns an empty dict
        # because it's in include_outputs_from
        assert "empty_processor" in result
        assert result["empty_processor"] == {}
