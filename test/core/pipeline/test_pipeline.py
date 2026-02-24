# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import logging
from concurrent.futures import ThreadPoolExecutor

import pytest

from haystack.components.joiners import BranchJoiner
from haystack.core.component import component
from haystack.core.errors import PipelineRuntimeError
from haystack.core.pipeline import Pipeline
from haystack.dataclasses.document import Document


@component
class WrongOutput:
    @component.output_types(output=str)
    def run(self, value: str) -> dict[str, str]:
        return "not_a_dict"  # type: ignore[return-value]


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
        pp = Pipeline()
        component_name = "joiner_1"
        pp.add_component(component_name, BranchJoiner(type_=str))
        pp.add_component("joiner_2", BranchJoiner(type_=str))
        pp.connect(component_name, "joiner_2")
        inputs = {"joiner_1": {"value": [{"sender": None, "value": "test_value"}]}}
        comp_dict = pp._get_component_with_graph_metadata_and_visits(component_name, 0)

        _ = pp._consume_component_inputs(component_name=component_name, component=comp_dict, inputs=inputs)
        # We remove input in greedy variadic sockets, even if they are from the user
        assert inputs == {"joiner_1": {}}

    def test__run_component_success(self):
        """Test successful component execution"""
        pp = Pipeline()
        component_name = "joiner_1"
        pp.add_component(component_name, BranchJoiner(type_=str))
        pp.add_component("joiner_2", BranchJoiner(type_=str))
        pp.connect(component_name, "joiner_2")

        outputs = pp._run_component(
            component_name=component_name,
            component=pp._get_component_with_graph_metadata_and_visits(component_name, 0),
            inputs={"value": ["test_value"]},
            component_visits={component_name: 0, "joiner_2": 0},
        )
        assert outputs == {"value": "test_value"}

    def test__run_component_fail(self):
        """Test error when component doesn't return a dictionary"""
        pp = Pipeline()
        pp.add_component("wrong", WrongOutput())

        with pytest.raises(PipelineRuntimeError) as exc_info:
            pp._run_component(
                component_name="wrong",
                component=pp._get_component_with_graph_metadata_and_visits("wrong", 0),
                inputs={"value": "test_value"},
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

        pp = Pipeline()
        pp.add_component("erroring_component", ErroringComponent())

        with pytest.raises(PipelineRuntimeError) as exc_info:
            pp._run_component(
                component_name="erroring_component",
                component=pp._get_component_with_graph_metadata_and_visits("erroring_component", 0),
                inputs={"wrong": {"value": [{"sender": None, "value": "test_value"}]}},
                component_visits={"erroring_component": 0},
            )
        assert "Component name: 'erroring_component'" in str(exc_info.value)

    def test_component_with_empty_dict_as_output_appears_in_results(self):
        """Test that components that return an empty dict as output appear in results as an empty dict"""

        @component
        class Producer:
            def __init__(self, prefix: str):
                self.prefix = prefix

            @component.output_types(value=str | None)
            def run(self, text: str | None) -> dict[str, str | None]:
                return {"value": f"{self.prefix}: {text}"}

        @component
        class EmptyProcessor:
            @component.output_types()
            def run(self, sources: list[str]) -> dict:
                # Returns empty dict when sources is empty
                return {}

        @component
        class Combiner:
            @component.output_types(combined=str)
            def run(self, input_a: str | None, input_b: str | None) -> dict[str, str]:
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

    def test_pipeline_is_possibly_blocked_warning_message(self, caplog):
        """
        Test that the pipeline raises a warning when it is possibly blocked due to missing inputs.

        The situation below looks a little contrived, but it has happened in practice that users create pipelines
        and accidentally made a mistake in their component code.
        """
        caplog.set_level(logging.WARNING)

        @component
        class MisconfiguredComponent:
            # Here we purposely declare other_output which is not actually returned by the run() method
            @component.output_types(output=str, other_output=str)
            def run(self, required_input: str) -> dict[str, str]:
                return {"output": "test"}

        @component
        class SimpleComponentTwoInputs:
            @component.output_types(output=str)
            def run(self, required_input: str, second_required_input: str) -> dict[str, str]:
                return {"output": "test"}

        pp = Pipeline()
        pp.add_component("first", MisconfiguredComponent())
        pp.add_component("second", SimpleComponentTwoInputs())

        # NOTE: We connect both outputs from the first component to the second component, but the first component
        # doesn't actually produce other_output, so the second component will be blocked due to missing input.
        pp.connect("first.output", "second.required_input")
        pp.connect("first.other_output", "second.second_required_input")

        pp.run({"first": {"required_input": "test"}})
        assert "Cannot run pipeline - the next component that is meant to run is blocked." in caplog.text
        assert "Component name: 'second'\nComponent type: 'SimpleComponentTwoInputs'" in caplog.text

    def test_pipeline_ensure_inputs_are_deep_copied(self):
        """
        Test to ensure that pipeline deep copies the inputs before passing them to components.

        This is important to prevent unintended side effects when components modify their inputs especially when
        the output from one component is passed to multiple other components.

        Some other notes about how this situation can arise in practice:
        - When a component returns a mutable object (like a Document) and that output is passed to multiple other
          components.
        - This doesn't happen when using output types like strings or integers, because they are not shared by
          reference so we will only commonly see this for objects like our dataclasses.
        """

        @component
        class SimpleComponent:
            @component.output_types(output=Document)
            def run(self, document: Document) -> dict[str, Document]:
                # Creates a new document to avoid modifying in place
                new_document = Document(content=document.content)
                return {"output": new_document}

        @component
        class ModifyingComponent:
            @component.output_types(output=Document)
            def run(self, document: Document) -> dict[str, Document]:
                # Modifies the incoming document inplace
                document.content = "modified"
                return {"output": document}

        pp = Pipeline()
        pp.add_component("first", SimpleComponent())
        pp.add_component("modifier", ModifyingComponent())
        # It's important that the following component has a name lower down the alphabetical order than "modifier",
        # since the pipeline runs components in a first-in-first-out manner based on ordered_component_names which is
        # sorted alphabetically.
        pp.add_component("second", SimpleComponent())

        pp.connect("first.output", "modifier.document")
        pp.connect("first.output", "second.document")

        result = pp.run({"first": {"document": Document(content="original")}})

        assert result["modifier"]["output"].content == "modified"
        # Without deep copying the inputs, the second component would also see the modified document and produce
        # "modified" instead of "original"
        assert result["second"]["output"].content == "original"
