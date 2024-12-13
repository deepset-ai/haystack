import json
from typing import List, Optional, Dict, Any
import re

from pytest_bdd import scenarios, given
import pytest

from haystack import Pipeline, Document, component
from haystack.document_stores.types import DuplicatePolicy
from haystack.dataclasses import ChatMessage, GeneratedAnswer
from haystack.components.routers import ConditionalRouter
from haystack.components.builders import PromptBuilder, AnswerBuilder, ChatPromptBuilder
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.joiners import BranchJoiner, DocumentJoiner, AnswerJoiner, StringJoiner
from haystack.testing.sample_components import (
    Accumulate,
    AddFixedValue,
    Double,
    Greet,
    Parity,
    Repeat,
    Subtract,
    Sum,
    Threshold,
    Remainder,
    FString,
    Hello,
    TextSplitter,
    StringListJoiner,
)
from haystack.testing.factory import component_class

from test.core.pipeline.features.conftest import PipelineRunData

pytestmark = pytest.mark.integration

scenarios("pipeline_run.feature")


@given("a pipeline that has no components", target_fixture="pipeline_data")
def pipeline_that_has_no_components():
    pipeline = Pipeline(max_runs_per_component=1)
    inputs = {}
    expected_outputs = {}
    return pipeline, [PipelineRunData(inputs=inputs, expected_outputs=expected_outputs)]


@given("a pipeline that is linear", target_fixture="pipeline_data")
def pipeline_that_is_linear():
    pipeline = Pipeline(max_runs_per_component=1)
    pipeline.add_component("first_addition", AddFixedValue(add=2))
    pipeline.add_component("second_addition", AddFixedValue())
    pipeline.add_component("double", Double())
    pipeline.connect("first_addition", "double")
    pipeline.connect("double", "second_addition")

    return (
        pipeline,
        [
            PipelineRunData(
                inputs={"first_addition": {"value": 1}},
                expected_outputs={"second_addition": {"result": 7}},
                expected_run_order=["first_addition", "double", "second_addition"],
            )
        ],
    )


@given("a pipeline that has an infinite loop", target_fixture="pipeline_data")
def pipeline_that_has_an_infinite_loop():
    routes = [
        {"condition": "{{number > 2}}", "output": "{{number}}", "output_name": "big_number", "output_type": int},
        {"condition": "{{number <= 2}}", "output": "{{number + 2}}", "output_name": "small_number", "output_type": int},
    ]

    main_input = BranchJoiner(int)
    first_router = ConditionalRouter(routes=routes)
    second_router = ConditionalRouter(routes=routes)

    pipe = Pipeline(max_runs_per_component=1)
    pipe.add_component("main_input", main_input)
    pipe.add_component("first_router", first_router)
    pipe.add_component("second_router", second_router)

    pipe.connect("main_input", "first_router.number")
    pipe.connect("first_router.big_number", "second_router.number")
    pipe.connect("second_router.big_number", "main_input")

    return pipe, [PipelineRunData({"main_input": {"value": 3}})]


@given("a pipeline that is really complex with lots of components, forks, and loops", target_fixture="pipeline_data")
def pipeline_complex():
    pipeline = Pipeline(max_runs_per_component=2)
    pipeline.add_component("greet_first", Greet(message="Hello, the value is {value}."))
    pipeline.add_component("accumulate_1", Accumulate())
    pipeline.add_component("add_two", AddFixedValue(add=2))
    pipeline.add_component("parity", Parity())
    pipeline.add_component("add_one", AddFixedValue(add=1))
    pipeline.add_component("accumulate_2", Accumulate())

    pipeline.add_component("branch_joiner", BranchJoiner(type_=int))
    pipeline.add_component("below_10", Threshold(threshold=10))
    pipeline.add_component("double", Double())

    pipeline.add_component("greet_again", Greet(message="Hello again, now the value is {value}."))
    pipeline.add_component("sum", Sum())

    pipeline.add_component("greet_enumerator", Greet(message="Hello from enumerator, here the value became {value}."))
    pipeline.add_component("enumerate", Repeat(outputs=["first", "second"]))
    pipeline.add_component("add_three", AddFixedValue(add=3))

    pipeline.add_component("diff", Subtract())
    pipeline.add_component("greet_one_last_time", Greet(message="Bye bye! The value here is {value}!"))
    pipeline.add_component("replicate", Repeat(outputs=["first", "second"]))
    pipeline.add_component("add_five", AddFixedValue(add=5))
    pipeline.add_component("add_four", AddFixedValue(add=4))
    pipeline.add_component("accumulate_3", Accumulate())

    pipeline.connect("greet_first", "accumulate_1")
    pipeline.connect("accumulate_1", "add_two")
    pipeline.connect("add_two", "parity")

    pipeline.connect("parity.even", "greet_again")
    pipeline.connect("greet_again", "sum.values")
    pipeline.connect("sum", "diff.first_value")
    pipeline.connect("diff", "greet_one_last_time")
    pipeline.connect("greet_one_last_time", "replicate")
    pipeline.connect("replicate.first", "add_five.value")
    pipeline.connect("replicate.second", "add_four.value")
    pipeline.connect("add_four", "accumulate_3")

    pipeline.connect("parity.odd", "add_one.value")
    pipeline.connect("add_one", "branch_joiner.value")
    pipeline.connect("branch_joiner", "below_10")

    pipeline.connect("below_10.below", "double")
    pipeline.connect("double", "branch_joiner.value")

    pipeline.connect("below_10.above", "accumulate_2")
    pipeline.connect("accumulate_2", "diff.second_value")

    pipeline.connect("greet_enumerator", "enumerate")
    pipeline.connect("enumerate.second", "sum.values")

    pipeline.connect("enumerate.first", "add_three.value")
    pipeline.connect("add_three", "sum.values")

    return (
        pipeline,
        [
            PipelineRunData(
                inputs={"greet_first": {"value": 1}, "greet_enumerator": {"value": 1}},
                expected_outputs={"accumulate_3": {"value": -7}, "add_five": {"result": -6}},
                expected_run_order=[
                    "greet_first",
                    "greet_enumerator",
                    "accumulate_1",
                    "enumerate",
                    "add_two",
                    "add_three",
                    "parity",
                    "add_one",
                    "branch_joiner",
                    "below_10",
                    "double",
                    "branch_joiner",
                    "below_10",
                    "double",
                    "branch_joiner",
                    "below_10",
                    "accumulate_2",
                    "sum",
                    "diff",
                    "greet_one_last_time",
                    "replicate",
                    "add_five",
                    "add_four",
                    "accumulate_3",
                ],
            )
        ],
    )


@given("a pipeline that has a single component with a default input", target_fixture="pipeline_data")
def pipeline_that_has_a_single_component_with_a_default_input():
    @component
    class WithDefault:
        @component.output_types(b=int)
        def run(self, a: int, b: int = 2):
            return {"c": a + b}

    pipeline = Pipeline(max_runs_per_component=1)
    pipeline.add_component("with_defaults", WithDefault())

    return (
        pipeline,
        [
            PipelineRunData(
                inputs={"with_defaults": {"a": 40, "b": 30}},
                expected_outputs={"with_defaults": {"c": 70}},
                expected_run_order=["with_defaults"],
            ),
            PipelineRunData(
                inputs={"with_defaults": {"a": 40}},
                expected_outputs={"with_defaults": {"c": 42}},
                expected_run_order=["with_defaults"],
            ),
        ],
    )


@given("a pipeline that has two loops of identical lengths", target_fixture="pipeline_data")
def pipeline_that_has_two_loops_of_identical_lengths():
    pipeline = Pipeline(max_runs_per_component=10)
    pipeline.add_component("branch_joiner", BranchJoiner(type_=int))
    pipeline.add_component("remainder", Remainder(divisor=3))
    pipeline.add_component("add_one", AddFixedValue(add=1))
    pipeline.add_component("add_two", AddFixedValue(add=2))

    pipeline.connect("branch_joiner.value", "remainder.value")
    pipeline.connect("remainder.remainder_is_1", "add_two.value")
    pipeline.connect("remainder.remainder_is_2", "add_one.value")
    pipeline.connect("add_two", "branch_joiner.value")
    pipeline.connect("add_one", "branch_joiner.value")
    return (
        pipeline,
        [
            PipelineRunData(
                inputs={"branch_joiner": {"value": 0}},
                expected_outputs={"remainder": {"remainder_is_0": 0}},
                expected_run_order=["branch_joiner", "remainder"],
            ),
            PipelineRunData(
                inputs={"branch_joiner": {"value": 3}},
                expected_outputs={"remainder": {"remainder_is_0": 3}},
                expected_run_order=["branch_joiner", "remainder"],
            ),
            PipelineRunData(
                inputs={"branch_joiner": {"value": 4}},
                expected_outputs={"remainder": {"remainder_is_0": 6}},
                expected_run_order=["branch_joiner", "remainder", "add_two", "branch_joiner", "remainder"],
            ),
            PipelineRunData(
                inputs={"branch_joiner": {"value": 5}},
                expected_outputs={"remainder": {"remainder_is_0": 6}},
                expected_run_order=["branch_joiner", "remainder", "add_one", "branch_joiner", "remainder"],
            ),
            PipelineRunData(
                inputs={"branch_joiner": {"value": 6}},
                expected_outputs={"remainder": {"remainder_is_0": 6}},
                expected_run_order=["branch_joiner", "remainder"],
            ),
        ],
    )


@given("a pipeline that has two loops of different lengths", target_fixture="pipeline_data")
def pipeline_that_has_two_loops_of_different_lengths():
    pipeline = Pipeline(max_runs_per_component=10)
    pipeline.add_component("branch_joiner", BranchJoiner(type_=int))
    pipeline.add_component("remainder", Remainder(divisor=3))
    pipeline.add_component("add_one", AddFixedValue(add=1))
    pipeline.add_component("add_two_1", AddFixedValue(add=1))
    pipeline.add_component("add_two_2", AddFixedValue(add=1))

    pipeline.connect("branch_joiner.value", "remainder.value")
    pipeline.connect("remainder.remainder_is_1", "add_two_1.value")
    pipeline.connect("add_two_1", "add_two_2.value")
    pipeline.connect("add_two_2", "branch_joiner")
    pipeline.connect("remainder.remainder_is_2", "add_one.value")
    pipeline.connect("add_one", "branch_joiner")

    return (
        pipeline,
        [
            PipelineRunData(
                inputs={"branch_joiner": {"value": 0}},
                expected_outputs={"remainder": {"remainder_is_0": 0}},
                expected_run_order=["branch_joiner", "remainder"],
            ),
            PipelineRunData(
                inputs={"branch_joiner": {"value": 3}},
                expected_outputs={"remainder": {"remainder_is_0": 3}},
                expected_run_order=["branch_joiner", "remainder"],
            ),
            PipelineRunData(
                inputs={"branch_joiner": {"value": 4}},
                expected_outputs={"remainder": {"remainder_is_0": 6}},
                expected_run_order=[
                    "branch_joiner",
                    "remainder",
                    "add_two_1",
                    "add_two_2",
                    "branch_joiner",
                    "remainder",
                ],
            ),
            PipelineRunData(
                inputs={"branch_joiner": {"value": 5}},
                expected_outputs={"remainder": {"remainder_is_0": 6}},
                expected_run_order=["branch_joiner", "remainder", "add_one", "branch_joiner", "remainder"],
            ),
            PipelineRunData(
                inputs={"branch_joiner": {"value": 6}},
                expected_outputs={"remainder": {"remainder_is_0": 6}},
                expected_run_order=["branch_joiner", "remainder"],
            ),
        ],
    )


@given("a pipeline that has a single loop with two conditional branches", target_fixture="pipeline_data")
def pipeline_that_has_a_single_loop_with_two_conditional_branches():
    accumulator = Accumulate()
    pipeline = Pipeline(max_runs_per_component=10)

    pipeline.add_component("add_one", AddFixedValue(add=1))
    pipeline.add_component("branch_joiner", BranchJoiner(type_=int))
    pipeline.add_component("below_10", Threshold(threshold=10))
    pipeline.add_component("below_5", Threshold(threshold=5))
    pipeline.add_component("add_three", AddFixedValue(add=3))
    pipeline.add_component("accumulator", accumulator)
    pipeline.add_component("add_two", AddFixedValue(add=2))

    pipeline.connect("add_one.result", "branch_joiner")
    pipeline.connect("branch_joiner.value", "below_10.value")
    pipeline.connect("below_10.below", "accumulator.value")
    pipeline.connect("accumulator.value", "below_5.value")
    pipeline.connect("below_5.above", "add_three.value")
    pipeline.connect("below_5.below", "branch_joiner")
    pipeline.connect("add_three.result", "branch_joiner")
    pipeline.connect("below_10.above", "add_two.value")

    return (
        pipeline,
        [
            PipelineRunData(
                inputs={"add_one": {"value": 3}},
                expected_outputs={"add_two": {"result": 13}},
                expected_run_order=[
                    "add_one",
                    "branch_joiner",
                    "below_10",
                    "accumulator",
                    "below_5",
                    "branch_joiner",
                    "below_10",
                    "accumulator",
                    "below_5",
                    "add_three",
                    "branch_joiner",
                    "below_10",
                    "add_two",
                ],
            )
        ],
    )


@given("a pipeline that has a component with dynamic inputs defined in init", target_fixture="pipeline_data")
def pipeline_that_has_a_component_with_dynamic_inputs_defined_in_init():
    pipeline = Pipeline(max_runs_per_component=1)
    pipeline.add_component("hello", Hello())
    pipeline.add_component("fstring", FString(template="This is the greeting: {greeting}!", variables=["greeting"]))
    pipeline.add_component("splitter", TextSplitter())
    pipeline.connect("hello.output", "fstring.greeting")
    pipeline.connect("fstring.string", "splitter.sentence")

    return (
        pipeline,
        [
            PipelineRunData(
                inputs={"hello": {"word": "Alice"}},
                expected_outputs={"splitter": {"output": ["This", "is", "the", "greeting:", "Hello,", "Alice!!"]}},
                expected_run_order=["hello", "fstring", "splitter"],
            ),
            PipelineRunData(
                inputs={"hello": {"word": "Alice"}, "fstring": {"template": "Received: {greeting}"}},
                expected_outputs={"splitter": {"output": ["Received:", "Hello,", "Alice!"]}},
                expected_run_order=["hello", "fstring", "splitter"],
            ),
        ],
    )


@given("a pipeline that has two branches that don't merge", target_fixture="pipeline_data")
def pipeline_that_has_two_branches_that_dont_merge():
    pipeline = Pipeline(max_runs_per_component=1)
    pipeline.add_component("add_one", AddFixedValue(add=1))
    pipeline.add_component("parity", Parity())
    pipeline.add_component("add_ten", AddFixedValue(add=10))
    pipeline.add_component("double", Double())
    pipeline.add_component("add_three", AddFixedValue(add=3))

    pipeline.connect("add_one.result", "parity.value")
    pipeline.connect("parity.even", "add_ten.value")
    pipeline.connect("parity.odd", "double.value")
    pipeline.connect("add_ten.result", "add_three.value")

    return (
        pipeline,
        [
            PipelineRunData(
                inputs={"add_one": {"value": 1}},
                expected_outputs={"add_three": {"result": 15}},
                expected_run_order=["add_one", "parity", "add_ten", "add_three"],
            ),
            PipelineRunData(
                inputs={"add_one": {"value": 2}},
                expected_outputs={"double": {"value": 6}},
                expected_run_order=["add_one", "parity", "double"],
            ),
        ],
    )


@given("a pipeline that has three branches that don't merge", target_fixture="pipeline_data")
def pipeline_that_has_three_branches_that_dont_merge():
    pipeline = Pipeline(max_runs_per_component=1)
    pipeline.add_component("add_one", AddFixedValue(add=1))
    pipeline.add_component("repeat", Repeat(outputs=["first", "second"]))
    pipeline.add_component("add_ten", AddFixedValue(add=10))
    pipeline.add_component("double", Double())
    pipeline.add_component("add_three", AddFixedValue(add=3))
    pipeline.add_component("add_one_again", AddFixedValue(add=1))

    pipeline.connect("add_one.result", "repeat.value")
    pipeline.connect("repeat.first", "add_ten.value")
    pipeline.connect("repeat.second", "double.value")
    pipeline.connect("repeat.second", "add_three.value")
    pipeline.connect("add_three.result", "add_one_again.value")

    return (
        pipeline,
        [
            PipelineRunData(
                inputs={"add_one": {"value": 1}},
                expected_outputs={"add_one_again": {"result": 6}, "add_ten": {"result": 12}, "double": {"value": 4}},
                expected_run_order=["add_one", "repeat", "add_ten", "double", "add_three", "add_one_again"],
            )
        ],
    )


@given("a pipeline that has two branches that merge", target_fixture="pipeline_data")
def pipeline_that_has_two_branches_that_merge():
    pipeline = Pipeline(max_runs_per_component=1)
    pipeline.add_component("first_addition", AddFixedValue(add=2))
    pipeline.add_component("second_addition", AddFixedValue(add=2))
    pipeline.add_component("third_addition", AddFixedValue(add=2))
    pipeline.add_component("diff", Subtract())
    pipeline.add_component("fourth_addition", AddFixedValue(add=1))

    pipeline.connect("first_addition.result", "second_addition.value")
    pipeline.connect("second_addition.result", "diff.first_value")
    pipeline.connect("third_addition.result", "diff.second_value")
    pipeline.connect("diff", "fourth_addition.value")
    return (
        pipeline,
        [
            PipelineRunData(
                inputs={"first_addition": {"value": 1}, "third_addition": {"value": 1}},
                expected_outputs={"fourth_addition": {"result": 3}},
                expected_run_order=["first_addition", "third_addition", "second_addition", "diff", "fourth_addition"],
            )
        ],
    )


@given(
    "a pipeline that has different combinations of branches that merge and do not merge", target_fixture="pipeline_data"
)
def pipeline_that_has_different_combinations_of_branches_that_merge_and_do_not_merge():
    pipeline = Pipeline(max_runs_per_component=1)
    pipeline.add_component("add_one", AddFixedValue())
    pipeline.add_component("parity", Parity())
    pipeline.add_component("add_ten", AddFixedValue(add=10))
    pipeline.add_component("double", Double())
    pipeline.add_component("add_four", AddFixedValue(add=4))
    pipeline.add_component("add_two", AddFixedValue())
    pipeline.add_component("add_two_as_well", AddFixedValue())
    pipeline.add_component("diff", Subtract())

    pipeline.connect("add_one.result", "parity.value")
    pipeline.connect("parity.even", "add_four.value")
    pipeline.connect("parity.odd", "double.value")
    pipeline.connect("add_ten.result", "diff.first_value")
    pipeline.connect("double.value", "diff.second_value")
    pipeline.connect("parity.odd", "add_ten.value")
    pipeline.connect("add_four.result", "add_two.value")
    pipeline.connect("add_four.result", "add_two_as_well.value")

    return (
        pipeline,
        [
            PipelineRunData(
                inputs={"add_one": {"value": 1}, "add_two": {"add": 2}, "add_two_as_well": {"add": 2}},
                expected_outputs={"add_two": {"result": 8}, "add_two_as_well": {"result": 8}},
                expected_run_order=["add_one", "parity", "add_four", "add_two", "add_two_as_well"],
            ),
            PipelineRunData(
                inputs={"add_one": {"value": 2}, "add_two": {"add": 2}, "add_two_as_well": {"add": 2}},
                expected_outputs={"diff": {"difference": 7}},
                expected_run_order=["add_one", "parity", "double", "add_ten", "diff"],
            ),
        ],
    )


@given("a pipeline that has two branches, one of which loops back", target_fixture="pipeline_data")
def pipeline_that_has_two_branches_one_of_which_loops_back():
    pipeline = Pipeline(max_runs_per_component=10)
    pipeline.add_component("add_zero", AddFixedValue(add=0))
    pipeline.add_component("branch_joiner", BranchJoiner(type_=int))
    pipeline.add_component("sum", Sum())
    pipeline.add_component("below_10", Threshold(threshold=10))
    pipeline.add_component("add_one", AddFixedValue(add=1))
    pipeline.add_component("counter", Accumulate())
    pipeline.add_component("add_two", AddFixedValue(add=2))

    pipeline.connect("add_zero", "branch_joiner.value")
    pipeline.connect("branch_joiner", "below_10.value")
    pipeline.connect("below_10.below", "add_one.value")
    pipeline.connect("add_one.result", "counter.value")
    pipeline.connect("counter.value", "branch_joiner.value")
    pipeline.connect("below_10.above", "add_two.value")
    pipeline.connect("add_two.result", "sum.values")

    return (
        pipeline,
        [
            PipelineRunData(
                inputs={"add_zero": {"value": 8}, "sum": {"values": 2}},
                expected_outputs={"sum": {"total": 23}},
                expected_run_order=[
                    "add_zero",
                    "branch_joiner",
                    "below_10",
                    "add_one",
                    "counter",
                    "branch_joiner",
                    "below_10",
                    "add_one",
                    "counter",
                    "branch_joiner",
                    "below_10",
                    "add_two",
                    "sum",
                ],
            )
        ],
    )


@given("a pipeline that has a component with mutable input", target_fixture="pipeline_data")
def pipeline_that_has_a_component_with_mutable_input():
    @component
    class InputMangler:
        @component.output_types(mangled_list=List[str])
        def run(self, input_list: List[str]):
            input_list.append("extra_item")
            return {"mangled_list": input_list}

    pipe = Pipeline(max_runs_per_component=1)
    pipe.add_component("mangler1", InputMangler())
    pipe.add_component("mangler2", InputMangler())
    pipe.add_component("concat1", StringListJoiner())
    pipe.add_component("concat2", StringListJoiner())
    pipe.connect("mangler1", "concat1")
    pipe.connect("mangler2", "concat2")

    input_list = ["foo", "bar"]

    return (
        pipe,
        [
            PipelineRunData(
                inputs={"mangler1": {"input_list": input_list}, "mangler2": {"input_list": input_list}},
                expected_outputs={
                    "concat1": {"output": ["foo", "bar", "extra_item"]},
                    "concat2": {"output": ["foo", "bar", "extra_item"]},
                },
                expected_run_order=["mangler1", "mangler2", "concat1", "concat2"],
            )
        ],
    )


@given("a pipeline that has a component with mutable output sent to multiple inputs", target_fixture="pipeline_data")
def pipeline_that_has_a_component_with_mutable_output_sent_to_multiple_inputs():
    @component
    class PassThroughPromptBuilder:
        # This is a pass-through component that returns the same input
        @component.output_types(prompt=List[ChatMessage])
        def run(self, prompt_source: List[ChatMessage]):
            return {"prompt": prompt_source}

    @component
    class MessageMerger:
        @component.output_types(merged_message=str)
        def run(self, messages: List[ChatMessage], metadata: dict = None):
            return {"merged_message": "\n".join(t.text or "" for t in messages)}

    @component
    class FakeGenerator:
        # This component is a fake generator that always returns the same message
        @component.output_types(replies=List[ChatMessage])
        def run(self, messages: List[ChatMessage]):
            return {"replies": [ChatMessage.from_assistant("Fake message")]}

    prompt_builder = PassThroughPromptBuilder()
    llm = FakeGenerator()
    mm1 = MessageMerger()
    mm2 = MessageMerger()

    pipe = Pipeline(max_runs_per_component=1)
    pipe.add_component("prompt_builder", prompt_builder)
    pipe.add_component("llm", llm)
    pipe.add_component("mm1", mm1)
    pipe.add_component("mm2", mm2)

    pipe.connect("prompt_builder.prompt", "llm.messages")
    pipe.connect("prompt_builder.prompt", "mm1")
    pipe.connect("llm.replies", "mm2")

    messages = [
        ChatMessage.from_system("Always respond in English even if some input data is in other languages."),
        ChatMessage.from_user("Tell me about Berlin"),
    ]
    params = {"metadata": {"metadata_key": "metadata_value", "meta2": "value2"}}

    return (
        pipe,
        [
            PipelineRunData(
                inputs={"mm1": params, "mm2": params, "prompt_builder": {"prompt_source": messages}},
                expected_outputs={
                    "mm1": {
                        "merged_message": "Always respond "
                        "in English even "
                        "if some input "
                        "data is in other "
                        "languages.\n"
                        "Tell me about "
                        "Berlin"
                    },
                    "mm2": {"merged_message": "Fake message"},
                },
                expected_run_order=["prompt_builder", "llm", "mm1", "mm2"],
            )
        ],
    )


@given(
    "a pipeline that has a greedy and variadic component after a component with default input",
    target_fixture="pipeline_data",
)
def pipeline_that_has_a_greedy_and_variadic_component_after_a_component_with_default_input():
    """
    This test verifies that `Pipeline.run()` executes the components in the correct order when
    there's a greedy Component with variadic input right before a Component with at least one default input.

    We use the `spying_tracer` fixture to simplify the code to verify the order of execution.
    This creates some coupling between this test and how we trace the Pipeline execution.
    A worthy tradeoff in my opinion, we will notice right away if we change either the run logic or
    the tracing logic.
    """
    document_store = InMemoryDocumentStore()
    document_store.write_documents([Document(content="This is a simple document")])

    pipeline = Pipeline(max_runs_per_component=1)
    template = "Given this documents: {{ documents|join(', ', attribute='content') }} Answer this question: {{ query }}"
    pipeline.add_component("retriever", InMemoryBM25Retriever(document_store=document_store))
    pipeline.add_component("prompt_builder", PromptBuilder(template=template))
    pipeline.add_component("branch_joiner", BranchJoiner(List[Document]))

    pipeline.connect("retriever", "branch_joiner")
    pipeline.connect("branch_joiner", "prompt_builder.documents")
    return (
        pipeline,
        [
            PipelineRunData(
                inputs={"query": "This is my question"},
                expected_outputs={
                    "prompt_builder": {
                        "prompt": "Given this "
                        "documents: "
                        "This is a "
                        "simple "
                        "document "
                        "Answer this "
                        "question: "
                        "This is my "
                        "question"
                    }
                },
                expected_run_order=["retriever", "branch_joiner", "prompt_builder"],
            )
        ],
    )


@given("a pipeline that has a component that doesn't return a dictionary", target_fixture="pipeline_data")
def pipeline_that_has_a_component_that_doesnt_return_a_dictionary():
    BrokenComponent = component_class(
        "BrokenComponent",
        input_types={"a": int},
        output_types={"b": int},
        output=1,  # type:ignore
    )

    pipe = Pipeline(max_runs_per_component=10)
    pipe.add_component("comp", BrokenComponent())
    return pipe, [PipelineRunData({"comp": {"a": 1}})]


@given(
    "a pipeline that has components added in a different order from the order of execution",
    target_fixture="pipeline_data",
)
def pipeline_that_has_components_added_in_a_different_order_from_the_order_of_execution():
    """
    We enqueue the Components in internal `to_run` data structure at the start of `Pipeline.run()` using the order
    they are added in the Pipeline with `Pipeline.add_component()`.
    If a Component A with defaults is added before a Component B that has no defaults, but in the Pipeline
    logic A must be executed after B it could run instead before.

    This test verifies that the order of execution is correct.
    """
    docs = [Document(content="Rome is the capital of Italy"), Document(content="Paris is the capital of France")]
    doc_store = InMemoryDocumentStore()
    doc_store.write_documents(docs)
    template = (
        "Given the following information, answer the question.\n"
        "Context:\n"
        "{% for document in documents %}"
        "    {{ document.content }}\n"
        "{% endfor %}"
        "Question: {{ query }}"
    )

    pipe = Pipeline(max_runs_per_component=1)

    # The order of this addition is important for the test
    # Do not edit them.
    pipe.add_component("prompt_builder", PromptBuilder(template=template))
    pipe.add_component("retriever", InMemoryBM25Retriever(document_store=doc_store))
    pipe.connect("retriever", "prompt_builder.documents")

    query = "What is the capital of France?"
    return (
        pipe,
        [
            PipelineRunData(
                inputs={"prompt_builder": {"query": query}, "retriever": {"query": query}},
                expected_outputs={
                    "prompt_builder": {
                        "prompt": "Given the "
                        "following "
                        "information, "
                        "answer the "
                        "question.\n"
                        "Context:\n"
                        "    Paris is "
                        "the capital "
                        "of France\n"
                        "    Rome is "
                        "the capital "
                        "of Italy\n"
                        "Question: "
                        "What is the "
                        "capital of "
                        "France?"
                    }
                },
                expected_run_order=["retriever", "prompt_builder"],
            )
        ],
    )


@given("a pipeline that has a component with only default inputs", target_fixture="pipeline_data")
def pipeline_that_has_a_component_with_only_default_inputs():
    FakeGenerator = component_class(
        "FakeGenerator", input_types={"prompt": str}, output_types={"replies": List[str]}, output={"replies": ["Paris"]}
    )
    docs = [Document(content="Rome is the capital of Italy"), Document(content="Paris is the capital of France")]
    doc_store = InMemoryDocumentStore()
    doc_store.write_documents(docs)
    template = (
        "Given the following information, answer the question.\n"
        "Context:\n"
        "{% for document in documents %}"
        "    {{ document.content }}\n"
        "{% endfor %}"
        "Question: {{ query }}"
    )

    pipe = Pipeline(max_runs_per_component=1)

    pipe.add_component("retriever", InMemoryBM25Retriever(document_store=doc_store))
    pipe.add_component("prompt_builder", PromptBuilder(template=template))
    pipe.add_component("generator", FakeGenerator())
    pipe.add_component("answer_builder", AnswerBuilder())

    pipe.connect("retriever", "prompt_builder.documents")
    pipe.connect("prompt_builder.prompt", "generator.prompt")
    pipe.connect("generator.replies", "answer_builder.replies")
    pipe.connect("retriever.documents", "answer_builder.documents")

    return (
        pipe,
        [
            PipelineRunData(
                inputs={"query": "What is the capital of France?"},
                expected_outputs={
                    "answer_builder": {
                        "answers": [
                            GeneratedAnswer(
                                data="Paris",
                                query="What " "is " "the " "capital " "of " "France?",
                                documents=[
                                    Document(
                                        id="413dccdf51a54cca75b7ed2eddac04e6e58560bd2f0caf4106a3efc023fe3651",
                                        content="Paris is the capital of France",
                                        score=1.600237583702734,
                                    ),
                                    Document(
                                        id="a4a874fc2ef75015da7924d709fbdd2430e46a8e94add6e0f26cd32c1c03435d",
                                        content="Rome is the capital of Italy",
                                        score=1.2536639934227616,
                                    ),
                                ],
                                meta={},
                            )
                        ]
                    }
                },
                expected_run_order=["retriever", "prompt_builder", "generator", "answer_builder"],
            )
        ],
    )


@given(
    "a pipeline that has a component with only default inputs as first to run and receives inputs from a loop",
    target_fixture="pipeline_data",
)
def pipeline_that_has_a_component_with_only_default_inputs_as_first_to_run_and_receives_inputs_from_a_loop():
    """
    This tests verifies that a Pipeline doesn't get stuck running in a loop if
    it has all the following characterics:
    - The first Component has all defaults for its inputs
    - The first Component receives one input from the user
    - The first Component receives one input from a loop in the Pipeline
    - The second Component has at least one default input
    """

    def fake_generator_run(self, generation_kwargs: Optional[Dict[str, Any]] = None, **kwargs):
        # Simple hack to simulate a model returning a different reply after the
        # the first time it's called
        if getattr(fake_generator_run, "called", False):
            return {"replies": ["Rome"]}
        fake_generator_run.called = True
        return {"replies": ["Paris"]}

    FakeGenerator = component_class(
        "FakeGenerator",
        input_types={"prompt": str},
        output_types={"replies": List[str]},
        extra_fields={"run": fake_generator_run},
    )
    template = (
        "Answer the following question.\n"
        "{% if previous_replies %}\n"
        "Previously you replied incorrectly this:\n"
        "{% for reply in previous_replies %}\n"
        " - {{ reply }}\n"
        "{% endfor %}\n"
        "{% endif %}\n"
        "Question: {{ query }}"
    )
    router = ConditionalRouter(
        routes=[
            {
                "condition": "{{ replies == ['Rome'] }}",
                "output": "{{ replies }}",
                "output_name": "correct_replies",
                "output_type": List[int],
            },
            {
                "condition": "{{ replies == ['Paris'] }}",
                "output": "{{ replies }}",
                "output_name": "incorrect_replies",
                "output_type": List[int],
            },
        ]
    )

    pipe = Pipeline(max_runs_per_component=1)

    pipe.add_component("prompt_builder", PromptBuilder(template=template))
    pipe.add_component("generator", FakeGenerator())
    pipe.add_component("router", router)

    pipe.connect("prompt_builder.prompt", "generator.prompt")
    pipe.connect("generator.replies", "router.replies")
    pipe.connect("router.incorrect_replies", "prompt_builder.previous_replies")

    return (
        pipe,
        [
            PipelineRunData(
                inputs={"prompt_builder": {"query": "What is the capital of " "Italy?"}},
                expected_outputs={"router": {"correct_replies": ["Rome"]}},
                expected_run_order=["prompt_builder", "generator", "router", "prompt_builder", "generator", "router"],
            )
        ],
    )


@given(
    "a pipeline that has multiple branches that merge into a component with a single variadic input",
    target_fixture="pipeline_data",
)
def pipeline_that_has_multiple_branches_that_merge_into_a_component_with_a_single_variadic_input():
    pipeline = Pipeline(max_runs_per_component=1)
    pipeline.add_component("add_one", AddFixedValue())
    pipeline.add_component("parity", Remainder(divisor=2))
    pipeline.add_component("add_ten", AddFixedValue(add=10))
    pipeline.add_component("double", Double())
    pipeline.add_component("add_four", AddFixedValue(add=4))
    pipeline.add_component("add_one_again", AddFixedValue())
    pipeline.add_component("sum", Sum())

    pipeline.connect("add_one.result", "parity.value")
    pipeline.connect("parity.remainder_is_0", "add_ten.value")
    pipeline.connect("parity.remainder_is_1", "double.value")
    pipeline.connect("add_one.result", "sum.values")
    pipeline.connect("add_ten.result", "sum.values")
    pipeline.connect("double.value", "sum.values")
    pipeline.connect("parity.remainder_is_1", "add_four.value")
    pipeline.connect("add_four.result", "add_one_again.value")
    pipeline.connect("add_one_again.result", "sum.values")

    return (
        pipeline,
        [
            PipelineRunData(
                inputs={"add_one": {"value": 1}},
                expected_outputs={"sum": {"total": 14}},
                expected_run_order=["add_one", "parity", "add_ten", "sum"],
            ),
            PipelineRunData(
                inputs={"add_one": {"value": 2}},
                expected_outputs={"sum": {"total": 17}},
                expected_run_order=["add_one", "parity", "double", "add_four", "add_one_again", "sum"],
            ),
        ],
    )


@given(
    "a pipeline that has multiple branches of different lengths that merge into a component with a single variadic input",
    target_fixture="pipeline_data",
)
def pipeline_that_has_multiple_branches_of_different_lengths_that_merge_into_a_component_with_a_single_variadic_input():
    pipeline = Pipeline(max_runs_per_component=1)
    pipeline.add_component("first_addition", AddFixedValue(add=2))
    pipeline.add_component("second_addition", AddFixedValue(add=2))
    pipeline.add_component("third_addition", AddFixedValue(add=2))
    pipeline.add_component("sum", Sum())
    pipeline.add_component("fourth_addition", AddFixedValue(add=1))

    pipeline.connect("first_addition.result", "second_addition.value")
    pipeline.connect("first_addition.result", "sum.values")
    pipeline.connect("second_addition.result", "sum.values")
    pipeline.connect("third_addition.result", "sum.values")
    pipeline.connect("sum.total", "fourth_addition.value")

    return (
        pipeline,
        [
            PipelineRunData(
                inputs={"first_addition": {"value": 1}, "third_addition": {"value": 1}},
                expected_outputs={"fourth_addition": {"result": 12}},
                expected_run_order=["first_addition", "third_addition", "second_addition", "sum", "fourth_addition"],
            )
        ],
    )


@given("a pipeline that is linear and returns intermediate outputs", target_fixture="pipeline_data")
def pipeline_that_is_linear_and_returns_intermediate_outputs():
    pipeline = Pipeline(max_runs_per_component=1)
    pipeline.add_component("first_addition", AddFixedValue(add=2))
    pipeline.add_component("second_addition", AddFixedValue())
    pipeline.add_component("double", Double())
    pipeline.connect("first_addition", "double")
    pipeline.connect("double", "second_addition")

    return (
        pipeline,
        [
            PipelineRunData(
                inputs={"first_addition": {"value": 1}},
                include_outputs_from={"second_addition", "double", "first_addition"},
                expected_outputs={
                    "double": {"value": 6},
                    "first_addition": {"result": 3},
                    "second_addition": {"result": 7},
                },
                expected_run_order=["first_addition", "double", "second_addition"],
            ),
            PipelineRunData(
                inputs={"first_addition": {"value": 1}},
                include_outputs_from={"double"},
                expected_outputs={"double": {"value": 6}, "second_addition": {"result": 7}},
                expected_run_order=["first_addition", "double", "second_addition"],
            ),
        ],
    )


@given("a pipeline that has a loop and returns intermediate outputs from it", target_fixture="pipeline_data")
def pipeline_that_has_a_loop_and_returns_intermediate_outputs_from_it():
    pipeline = Pipeline(max_runs_per_component=10)
    pipeline.add_component("add_one", AddFixedValue(add=1))
    pipeline.add_component("branch_joiner", BranchJoiner(type_=int))
    pipeline.add_component("below_10", Threshold(threshold=10))
    pipeline.add_component("below_5", Threshold(threshold=5))
    pipeline.add_component("add_three", AddFixedValue(add=3))
    pipeline.add_component("accumulator", Accumulate())
    pipeline.add_component("add_two", AddFixedValue(add=2))

    pipeline.connect("add_one.result", "branch_joiner")
    pipeline.connect("branch_joiner.value", "below_10.value")
    pipeline.connect("below_10.below", "accumulator.value")
    pipeline.connect("accumulator.value", "below_5.value")
    pipeline.connect("below_5.above", "add_three.value")
    pipeline.connect("below_5.below", "branch_joiner")
    pipeline.connect("add_three.result", "branch_joiner")
    pipeline.connect("below_10.above", "add_two.value")

    return (
        pipeline,
        [
            PipelineRunData(
                inputs={"add_one": {"value": 3}},
                include_outputs_from={
                    "add_two",
                    "add_one",
                    "branch_joiner",
                    "below_10",
                    "accumulator",
                    "below_5",
                    "add_three",
                },
                expected_outputs={
                    "add_two": {"result": 13},
                    "add_one": {"result": 4},
                    "branch_joiner": {"value": 11},
                    "below_10": {"above": 11},
                    "accumulator": {"value": 8},
                    "below_5": {"above": 8},
                    "add_three": {"result": 11},
                },
                expected_run_order=[
                    "add_one",
                    "branch_joiner",
                    "below_10",
                    "accumulator",
                    "below_5",
                    "branch_joiner",
                    "below_10",
                    "accumulator",
                    "below_5",
                    "add_three",
                    "branch_joiner",
                    "below_10",
                    "add_two",
                ],
            )
        ],
    )


@given(
    "a pipeline that is linear and returns intermediate outputs from multiple sockets", target_fixture="pipeline_data"
)
def pipeline_that_is_linear_and_returns_intermediate_outputs_from_multiple_sockets():
    @component
    class DoubleWithOriginal:
        """
        Doubles the input value and returns the original value as well.
        """

        @component.output_types(value=int, original=int)
        def run(self, value: int):
            return {"value": value * 2, "original": value}

    pipeline = Pipeline(max_runs_per_component=1)
    pipeline.add_component("first_addition", AddFixedValue(add=2))
    pipeline.add_component("second_addition", AddFixedValue())
    pipeline.add_component("double", DoubleWithOriginal())
    pipeline.connect("first_addition", "double")
    pipeline.connect("double.value", "second_addition")

    return (
        pipeline,
        [
            PipelineRunData(
                inputs={"first_addition": {"value": 1}},
                include_outputs_from={"second_addition", "double", "first_addition"},
                expected_outputs={
                    "double": {"original": 3, "value": 6},
                    "first_addition": {"result": 3},
                    "second_addition": {"result": 7},
                },
                expected_run_order=["first_addition", "double", "second_addition"],
            ),
            PipelineRunData(
                inputs={"first_addition": {"value": 1}},
                include_outputs_from={"double"},
                expected_outputs={"double": {"original": 3, "value": 6}, "second_addition": {"result": 7}},
                expected_run_order=["first_addition", "double", "second_addition"],
            ),
        ],
    )


@given(
    "a pipeline that has a component with default inputs that doesn't receive anything from its sender",
    target_fixture="pipeline_data",
)
def pipeline_that_has_a_component_with_default_inputs_that_doesnt_receive_anything_from_its_sender():
    routes = [
        {"condition": "{{'reisen' in sentence}}", "output": "German", "output_name": "language_1", "output_type": str},
        {"condition": "{{'viajar' in sentence}}", "output": "Spanish", "output_name": "language_2", "output_type": str},
    ]
    router = ConditionalRouter(routes)

    pipeline = Pipeline(max_runs_per_component=1)
    pipeline.add_component("router", router)
    pipeline.add_component("pb", PromptBuilder(template="Ok, I know, that's {{language}}"))
    pipeline.connect("router.language_2", "pb.language")

    return (
        pipeline,
        [
            PipelineRunData(
                inputs={"router": {"sentence": "Wir mussen reisen"}},
                expected_outputs={"router": {"language_1": "German"}},
                expected_run_order=["router"],
            ),
            PipelineRunData(
                inputs={"router": {"sentence": "Yo tengo que viajar"}},
                expected_outputs={"pb": {"prompt": "Ok, I know, that's Spanish"}},
                expected_run_order=["router", "pb"],
            ),
        ],
    )


@given(
    "a pipeline that has a component with default inputs that doesn't receive anything from its sender but receives input from user",
    target_fixture="pipeline_data",
)
def pipeline_that_has_a_component_with_default_inputs_that_doesnt_receive_anything_from_its_sender_but_receives_input_from_user():
    prompt = PromptBuilder(
        template="""Please generate an SQL query. The query should answer the following Question: {{ question }};
            If the question cannot be answered given the provided table and columns, return 'no_answer'
            The query is to be answered for the table is called 'absenteeism' with the following
            Columns: {{ columns }};
            Answer:"""
    )

    @component
    class FakeGenerator:
        @component.output_types(replies=List[str])
        def run(self, prompt: str):
            if "no_answer" in prompt:
                return {"replies": ["There's simply no_answer to this question"]}
            return {"replies": ["Some SQL query"]}

    @component
    class FakeSQLQuerier:
        @component.output_types(results=str)
        def run(self, query: str):
            return {"results": "This is the query result", "query": query}

    llm = FakeGenerator()
    sql_querier = FakeSQLQuerier()

    routes = [
        {
            "condition": "{{'no_answer' not in replies[0]}}",
            "output": "{{replies[0]}}",
            "output_name": "sql",
            "output_type": str,
        },
        {
            "condition": "{{'no_answer' in replies[0]}}",
            "output": "{{question}}",
            "output_name": "go_to_fallback",
            "output_type": str,
        },
    ]

    router = ConditionalRouter(routes)

    fallback_prompt = PromptBuilder(
        template="""User entered a query that cannot be answered with the given table.
                    The query was: {{ question }} and the table had columns: {{ columns }}.
                    Let the user know why the question cannot be answered"""
    )
    fallback_llm = FakeGenerator()

    pipeline = Pipeline(max_runs_per_component=1)
    pipeline.add_component("prompt", prompt)
    pipeline.add_component("llm", llm)
    pipeline.add_component("router", router)
    pipeline.add_component("fallback_prompt", fallback_prompt)
    pipeline.add_component("fallback_llm", fallback_llm)
    pipeline.add_component("sql_querier", sql_querier)

    pipeline.connect("prompt", "llm")
    pipeline.connect("llm.replies", "router.replies")
    pipeline.connect("router.sql", "sql_querier.query")
    pipeline.connect("router.go_to_fallback", "fallback_prompt.question")
    pipeline.connect("fallback_prompt", "fallback_llm")

    columns = "Age, Absenteeism_time_in_hours, Days, Disciplinary_failure"
    return (
        pipeline,
        [
            PipelineRunData(
                inputs={
                    "prompt": {"question": "This is a question with no_answer", "columns": columns},
                    "router": {"question": "This is a question with no_answer"},
                },
                expected_outputs={"fallback_llm": {"replies": ["There's simply no_answer to this question"]}},
                expected_run_order=["prompt", "llm", "router", "fallback_prompt", "fallback_llm"],
            )
        ],
        [
            PipelineRunData(
                inputs={
                    "prompt": {"question": "This is a question that has an answer", "columns": columns},
                    "router": {"question": "This is a question that has an answer"},
                },
                expected_outputs={"sql_querier": {"results": "This is the query result", "query": "Some SQL query"}},
                expected_run_order=["prompt", "llm", "router", "sql_querier"],
            )
        ],
    )


@given(
    "a pipeline that has a loop and a component with default inputs that doesn't receive anything from its sender but receives input from user",
    target_fixture="pipeline_data",
)
def pipeline_that_has_a_loop_and_a_component_with_default_inputs_that_doesnt_receive_anything_from_its_sender_but_receives_input_from_user():
    template = """
    You are an experienced and accurate Turkish CX speacialist that classifies customer comments into pre-defined categories below:\n
    Negative experience labels:
    - Late delivery
    - Rotten/spoilt item
    - Bad Courier behavior

    Positive experience labels:
    - Good courier behavior
    - Thanks & appreciation
    - Love message to courier
    - Fast delivery
    - Quality of products

    Create a JSON object as a response. The fields are: 'positive_experience', 'negative_experience'.
    Assign at least one of the pre-defined labels to the given customer comment under positive and negative experience fields.
    If the comment has a positive experience, list the label under 'positive_experience' field.
    If the comments has a negative_experience, list it under the 'negative_experience' field.
    Here is the comment:\n{{ comment }}\n. Just return the category names in the list. If there aren't any, return an empty list.

    {% if invalid_replies and error_message %}
    You already created the following output in a previous attempt: {{ invalid_replies }}
    However, this doesn't comply with the format requirements from above and triggered this Python exception: {{ error_message }}
    Correct the output and try again. Just return the corrected output without any extra explanations.
    {% endif %}
    """
    prompt_builder = PromptBuilder(template=template)

    @component
    class FakeOutputValidator:
        @component.output_types(
            valid_replies=List[str], invalid_replies=Optional[List[str]], error_message=Optional[str]
        )
        def run(self, replies: List[str]):
            if not getattr(self, "called", False):
                self.called = True
                return {"invalid_replies": ["This is an invalid reply"], "error_message": "this is an error message"}
            return {"valid_replies": replies}

    @component
    class FakeGenerator:
        @component.output_types(replies=List[str])
        def run(self, prompt: str):
            return {"replies": ["This is a valid reply"]}

    llm = FakeGenerator()
    validator = FakeOutputValidator()

    pipeline = Pipeline(max_runs_per_component=1)
    pipeline.add_component("prompt_builder", prompt_builder)

    pipeline.add_component("llm", llm)
    pipeline.add_component("output_validator", validator)

    pipeline.connect("prompt_builder.prompt", "llm.prompt")
    pipeline.connect("llm.replies", "output_validator.replies")
    pipeline.connect("output_validator.invalid_replies", "prompt_builder.invalid_replies")

    pipeline.connect("output_validator.error_message", "prompt_builder.error_message")

    comment = "I loved the quality of the meal but the courier was rude"
    return (
        pipeline,
        [
            PipelineRunData(
                inputs={"prompt_builder": {"template_variables": {"comment": comment}}},
                expected_outputs={"output_validator": {"valid_replies": ["This is a valid reply"]}},
                expected_run_order=[
                    "prompt_builder",
                    "llm",
                    "output_validator",
                    "prompt_builder",
                    "llm",
                    "output_validator",
                ],
            )
        ],
    )


@given(
    "a pipeline that has multiple components with only default inputs and are added in a different order from the order of execution",
    target_fixture="pipeline_data",
)
def pipeline_that_has_multiple_components_with_only_default_inputs_and_are_added_in_a_different_order_from_the_order_of_execution():
    prompt_builder1 = PromptBuilder(
        template="""
    You are a spellchecking system. Check the given query and fill in the corrected query.

    Question: {{question}}
    Corrected question:
    """
    )
    prompt_builder2 = PromptBuilder(
        template="""
    According to these documents:

    {% for doc in documents %}
    {{ doc.content }}
    {% endfor %}

    Answer the given question: {{question}}
    Answer:
    """
    )
    prompt_builder3 = PromptBuilder(
        template="""
    {% for ans in replies %}
    {{ ans }}
    {% endfor %}
    """
    )

    @component
    class FakeRetriever:
        @component.output_types(documents=List[Document])
        def run(
            self,
            query: str,
            filters: Optional[Dict[str, Any]] = None,
            top_k: Optional[int] = None,
            scale_score: Optional[bool] = None,
        ):
            return {"documents": [Document(content="This is a document")]}

    @component
    class FakeRanker:
        @component.output_types(documents=List[Document])
        def run(
            self,
            query: str,
            documents: List[Document],
            top_k: Optional[int] = None,
            scale_score: Optional[bool] = None,
            calibration_factor: Optional[float] = None,
            score_threshold: Optional[float] = None,
        ):
            return {"documents": documents}

    @component
    class FakeGenerator:
        @component.output_types(replies=List[str], meta=Dict[str, Any])
        def run(self, prompt: str, generation_kwargs: Optional[Dict[str, Any]] = None):
            return {"replies": ["This is a reply"], "meta": {"meta_key": "meta_value"}}

    pipeline = Pipeline(max_runs_per_component=1)
    pipeline.add_component(name="retriever", instance=FakeRetriever())
    pipeline.add_component(name="ranker", instance=FakeRanker())
    pipeline.add_component(name="prompt_builder2", instance=prompt_builder2)
    pipeline.add_component(name="prompt_builder1", instance=prompt_builder1)
    pipeline.add_component(name="prompt_builder3", instance=prompt_builder3)
    pipeline.add_component(name="llm", instance=FakeGenerator())
    pipeline.add_component(name="spellchecker", instance=FakeGenerator())

    pipeline.connect("prompt_builder1", "spellchecker")
    pipeline.connect("spellchecker.replies", "prompt_builder3")
    pipeline.connect("prompt_builder3", "retriever.query")
    pipeline.connect("prompt_builder3", "ranker.query")
    pipeline.connect("retriever.documents", "ranker.documents")
    pipeline.connect("ranker.documents", "prompt_builder2.documents")
    pipeline.connect("prompt_builder3", "prompt_builder2.question")
    pipeline.connect("prompt_builder2", "llm")

    return (
        pipeline,
        [
            PipelineRunData(
                inputs={"prompt_builder1": {"question": "Wha i Acromegaly?"}},
                expected_outputs={
                    "llm": {"replies": ["This is a reply"], "meta": {"meta_key": "meta_value"}},
                    "spellchecker": {"meta": {"meta_key": "meta_value"}},
                },
                expected_run_order=[
                    "prompt_builder1",
                    "spellchecker",
                    "prompt_builder3",
                    "retriever",
                    "ranker",
                    "prompt_builder2",
                    "llm",
                ],
            )
        ],
    )


@given("a pipeline that is linear with conditional branching and multiple joins", target_fixture="pipeline_data")
def that_is_linear_with_conditional_branching_and_multiple_joins():
    pipeline = Pipeline()

    @component
    class FakeRouter:
        @component.output_types(LEGIT=str, INJECTION=str)
        def run(self, query: str):
            if "injection" in query:
                return {"INJECTION": query}
            return {"LEGIT": query}

    @component
    class FakeEmbedder:
        @component.output_types(embeddings=List[float])
        def run(self, text: str):
            return {"embeddings": [1.0, 2.0, 3.0]}

    @component
    class FakeRanker:
        @component.output_types(documents=List[Document])
        def run(self, query: str, documents: List[Document]):
            return {"documents": documents}

    @component
    class FakeRetriever:
        @component.output_types(documents=List[Document])
        def run(self, query: str):
            if "injection" in query:
                return {"documents": []}
            return {"documents": [Document(content="This is a document")]}

    @component
    class FakeEmbeddingRetriever:
        @component.output_types(documents=List[Document])
        def run(self, query_embedding: List[float]):
            return {"documents": [Document(content="This is another document")]}

    pipeline.add_component(name="router", instance=FakeRouter())
    pipeline.add_component(name="text_embedder", instance=FakeEmbedder())
    pipeline.add_component(name="retriever", instance=FakeEmbeddingRetriever())
    pipeline.add_component(name="emptyretriever", instance=FakeRetriever())
    pipeline.add_component(name="joinerfinal", instance=DocumentJoiner())
    pipeline.add_component(name="joinerhybrid", instance=DocumentJoiner())
    pipeline.add_component(name="ranker", instance=FakeRanker())
    pipeline.add_component(name="bm25retriever", instance=FakeRetriever())

    pipeline.connect("router.INJECTION", "emptyretriever.query")
    pipeline.connect("router.LEGIT", "text_embedder.text")
    pipeline.connect("text_embedder", "retriever.query_embedding")
    pipeline.connect("router.LEGIT", "ranker.query")
    pipeline.connect("router.LEGIT", "bm25retriever.query")
    pipeline.connect("bm25retriever", "joinerhybrid.documents")
    pipeline.connect("retriever", "joinerhybrid.documents")
    pipeline.connect("joinerhybrid.documents", "ranker.documents")
    pipeline.connect("ranker", "joinerfinal.documents")
    pipeline.connect("emptyretriever", "joinerfinal.documents")

    return (
        pipeline,
        [
            PipelineRunData(
                inputs={"router": {"query": "I'm a legit question"}},
                expected_outputs={
                    "joinerfinal": {
                        "documents": [
                            Document(content="This is a document"),
                            Document(content="This is another document"),
                        ]
                    }
                },
                expected_run_order=[
                    "router",
                    "text_embedder",
                    "bm25retriever",
                    "retriever",
                    "joinerhybrid",
                    "ranker",
                    "joinerfinal",
                ],
            ),
            PipelineRunData(
                inputs={"router": {"query": "I'm a nasty prompt injection"}},
                expected_outputs={"joinerfinal": {"documents": []}},
                expected_run_order=["router", "emptyretriever", "joinerfinal"],
            ),
        ],
    )


@given("a pipeline that is a simple agent", target_fixture="pipeline_data")
def that_is_a_simple_agent():
    search_message_template = """
    Given these web search results:

    {% for doc in documents %}
        {{ doc.content }}
    {% endfor %}

    Be as brief as possible, max one sentence.
    Answer the question: {{search_query}}
    """

    react_message_template = """
    Solve a question answering task with interleaving Thought, Action, Observation steps.

    Thought reasons about the current situation

    Action can be:
    google_search - Searches Google for the exact concept/entity (given in square brackets) and returns the results for you to use
    finish - Returns the final answer (given in square brackets) and finishes the task

    Observation summarizes the Action outcome and helps in formulating the next
    Thought in Thought, Action, Observation interleaving triplet of steps.

    After each Observation, provide the next Thought and next Action.
    Don't execute multiple steps even though you know the answer.
    Only generate Thought and Action, never Observation, you'll get Observation from Action.
    Follow the pattern in the example below.

    Example:
    ###########################
    Question: Which magazine was started first Arthur’s Magazine or First for Women?
    Thought: I need to search Arthur’s Magazine and First for Women, and find which was started
    first.
    Action: google_search[When was 'Arthur’s Magazine' started?]
    Observation: Arthur’s Magazine was an American literary periodical ˘
    published in Philadelphia and founded in 1844. Edited by Timothy Shay Arthur, it featured work by
    Edgar A. Poe, J.H. Ingraham, Sarah Josepha Hale, Thomas G. Spear, and others. In May 1846
    it was merged into Godey’s Lady’s Book.
    Thought: Arthur’s Magazine was started in 1844. I need to search First for Women founding date next
    Action: google_search[When was 'First for Women' magazine started?]
    Observation: First for Women is a woman’s magazine published by Bauer Media Group in the
    USA. The magazine was started in 1989. It is based in Englewood Cliffs, New Jersey. In 2011
    the circulation of the magazine was 1,310,696 copies.
    Thought: First for Women was started in 1989. 1844 (Arthur’s Magazine) ¡ 1989 (First for
    Women), so Arthur’s Magazine was started first.
    Action: finish[Arthur’s Magazine]
    ############################

    Let's start, the question is: {{query}}

    Thought:
    """

    routes = [
        {
            "condition": "{{'search' in tool_id_and_param[0]}}",
            "output": "{{tool_id_and_param[1]}}",
            "output_name": "search",
            "output_type": str,
        },
        {
            "condition": "{{'finish' in tool_id_and_param[0]}}",
            "output": "{{tool_id_and_param[1]}}",
            "output_name": "finish",
            "output_type": str,
        },
    ]

    @component
    class FakeThoughtActionOpenAIChatGenerator:
        run_counter = 0

        @component.output_types(replies=List[ChatMessage])
        def run(self, messages: List[ChatMessage], generation_kwargs: Optional[Dict[str, Any]] = None):
            if self.run_counter == 0:
                self.run_counter += 1
                return {
                    "replies": [
                        ChatMessage.from_assistant(
                            "thinking\n Action: google_search[What is taller, Eiffel Tower or Leaning Tower of Pisa]\n"
                        )
                    ]
                }

            return {"replies": [ChatMessage.from_assistant("thinking\n Action: finish[Eiffel Tower]\n")]}

    @component
    class FakeConclusionOpenAIChatGenerator:
        @component.output_types(replies=List[ChatMessage])
        def run(self, messages: List[ChatMessage], generation_kwargs: Optional[Dict[str, Any]] = None):
            return {"replies": [ChatMessage.from_assistant("Tower of Pisa is 55 meters tall\n")]}

    @component
    class FakeSerperDevWebSearch:
        @component.output_types(documents=List[Document])
        def run(self, query: str):
            return {
                "documents": [
                    Document(content="Eiffel Tower is 300 meters tall"),
                    Document(content="Tower of Pisa is 55 meters tall"),
                ]
            }

    # main part
    pipeline = Pipeline()
    pipeline.add_component("main_input", BranchJoiner(List[ChatMessage]))
    pipeline.add_component("prompt_builder", ChatPromptBuilder(variables=["query"]))
    pipeline.add_component("llm", FakeThoughtActionOpenAIChatGenerator())

    @component
    class ToolExtractor:
        @component.output_types(output=List[str])
        def run(self, messages: List[ChatMessage]):
            prompt: str = messages[-1].text
            lines = prompt.strip().split("\n")
            for line in reversed(lines):
                pattern = r"Action:\s*(\w+)\[(.*?)\]"

                match = re.search(pattern, line)
                if match:
                    action_name = match.group(1)
                    parameter = match.group(2)
                    return {"output": [action_name, parameter]}
            return {"output": [None, None]}

    pipeline.add_component("tool_extractor", ToolExtractor())

    @component
    class PromptConcatenator:
        def __init__(self, suffix: str = ""):
            self._suffix = suffix

        @component.output_types(output=List[ChatMessage])
        def run(self, replies: List[ChatMessage], current_prompt: List[ChatMessage]):
            content = current_prompt[-1].text + replies[-1].text + self._suffix
            return {"output": [ChatMessage.from_user(content)]}

    @component
    class SearchOutputAdapter:
        @component.output_types(output=List[ChatMessage])
        def run(self, replies: List[ChatMessage]):
            content = f"Observation: {replies[-1].text}\n"
            return {"output": [ChatMessage.from_assistant(content)]}

    pipeline.add_component("prompt_concatenator_after_action", PromptConcatenator())

    pipeline.add_component("router", ConditionalRouter(routes))
    pipeline.add_component("router_search", FakeSerperDevWebSearch())
    pipeline.add_component("search_prompt_builder", ChatPromptBuilder(variables=["documents", "search_query"]))
    pipeline.add_component("search_llm", FakeConclusionOpenAIChatGenerator())

    pipeline.add_component("search_output_adapter", SearchOutputAdapter())
    pipeline.add_component("prompt_concatenator_after_observation", PromptConcatenator(suffix="\nThought: "))

    # main
    pipeline.connect("main_input", "prompt_builder.template")
    pipeline.connect("prompt_builder.prompt", "llm.messages")
    pipeline.connect("llm.replies", "prompt_concatenator_after_action.replies")

    # tools
    pipeline.connect("prompt_builder.prompt", "prompt_concatenator_after_action.current_prompt")
    pipeline.connect("prompt_concatenator_after_action", "tool_extractor.messages")

    pipeline.connect("tool_extractor", "router")
    pipeline.connect("router.search", "router_search.query")
    pipeline.connect("router_search.documents", "search_prompt_builder.documents")
    pipeline.connect("router.search", "search_prompt_builder.search_query")
    pipeline.connect("search_prompt_builder.prompt", "search_llm.messages")

    pipeline.connect("search_llm.replies", "search_output_adapter.replies")
    pipeline.connect("search_output_adapter", "prompt_concatenator_after_observation.replies")
    pipeline.connect("prompt_concatenator_after_action", "prompt_concatenator_after_observation.current_prompt")
    pipeline.connect("prompt_concatenator_after_observation", "main_input")

    search_message = [ChatMessage.from_user(search_message_template)]
    messages = [ChatMessage.from_user(react_message_template)]
    question = "which tower is taller: eiffel tower or tower of pisa?"

    return pipeline, [
        PipelineRunData(
            inputs={
                "main_input": {"value": messages},
                "prompt_builder": {"query": question},
                "search_prompt_builder": {"template": search_message},
            },
            expected_outputs={"router": {"finish": "Eiffel Tower"}},
            expected_run_order=[
                "main_input",
                "prompt_builder",
                "llm",
                "prompt_concatenator_after_action",
                "tool_extractor",
                "router",
                "router_search",
                "search_prompt_builder",
                "search_llm",
                "search_output_adapter",
                "prompt_concatenator_after_observation",
                "main_input",
                "prompt_builder",
                "llm",
                "prompt_concatenator_after_action",
                "tool_extractor",
                "router",
            ],
        )
    ]


@given("a pipeline that has a variadic component that receives partial inputs", target_fixture="pipeline_data")
def that_has_a_variadic_component_that_receives_partial_inputs():
    @component
    class ConditionalDocumentCreator:
        def __init__(self, content: str):
            self._content = content

        @component.output_types(documents=List[Document], noop=None)
        def run(self, create_document: bool = False):
            if create_document:
                return {"documents": [Document(id=self._content, content=self._content)]}
            return {"noop": None}

    pipeline = Pipeline(max_runs_per_component=1)
    pipeline.add_component("first_creator", ConditionalDocumentCreator(content="First document"))
    pipeline.add_component("second_creator", ConditionalDocumentCreator(content="Second document"))
    pipeline.add_component("third_creator", ConditionalDocumentCreator(content="Third document"))
    pipeline.add_component("documents_joiner", DocumentJoiner())

    pipeline.connect("first_creator.documents", "documents_joiner.documents")
    pipeline.connect("second_creator.documents", "documents_joiner.documents")
    pipeline.connect("third_creator.documents", "documents_joiner.documents")

    return (
        pipeline,
        [
            PipelineRunData(
                inputs={"first_creator": {"create_document": True}, "third_creator": {"create_document": True}},
                expected_outputs={
                    "second_creator": {"noop": None},
                    "documents_joiner": {
                        "documents": [
                            Document(id="First document", content="First document"),
                            Document(id="Third document", content="Third document"),
                        ]
                    },
                },
                expected_run_order=["first_creator", "second_creator", "third_creator", "documents_joiner"],
            ),
            PipelineRunData(
                inputs={"first_creator": {"create_document": True}, "second_creator": {"create_document": True}},
                expected_outputs={
                    "third_creator": {"noop": None},
                    "documents_joiner": {
                        "documents": [
                            Document(id="First document", content="First document"),
                            Document(id="Second document", content="Second document"),
                        ]
                    },
                },
                expected_run_order=["first_creator", "second_creator", "third_creator", "documents_joiner"],
            ),
        ],
    )


@given("a pipeline that has an answer joiner variadic component", target_fixture="pipeline_data")
def that_has_an_answer_joiner_variadic_component():
    query = "What's Natural Language Processing?"

    pipeline = Pipeline(max_runs_per_component=1)
    pipeline.add_component("answer_builder_1", AnswerBuilder())
    pipeline.add_component("answer_builder_2", AnswerBuilder())
    pipeline.add_component("answer_joiner", AnswerJoiner())

    pipeline.connect("answer_builder_1.answers", "answer_joiner")
    pipeline.connect("answer_builder_2.answers", "answer_joiner")

    return (
        pipeline,
        [
            PipelineRunData(
                inputs={
                    "answer_builder_1": {"query": query, "replies": ["This is a test answer"]},
                    "answer_builder_2": {"query": query, "replies": ["This is a second test answer"]},
                },
                expected_outputs={
                    "answer_joiner": {
                        "answers": [
                            GeneratedAnswer(
                                data="This is a test answer",
                                query="What's Natural Language Processing?",
                                documents=[],
                                meta={},
                            ),
                            GeneratedAnswer(
                                data="This is a second test answer",
                                query="What's Natural Language Processing?",
                                documents=[],
                                meta={},
                            ),
                        ]
                    }
                },
                expected_run_order=["answer_builder_1", "answer_builder_2", "answer_joiner"],
            )
        ],
    )


@given(
    "a pipeline that is linear and a component in the middle receives optional input from other components and input from the user",
    target_fixture="pipeline_data",
)
def that_is_linear_and_a_component_in_the_middle_receives_optional_input_from_other_components_and_input_from_the_user():
    @component
    class QueryMetadataExtractor:
        @component.output_types(filters=Dict[str, str])
        def run(self, prompt: str):
            metadata = json.loads(prompt)
            filters = []
            for key, value in metadata.items():
                filters.append({"field": f"meta.{key}", "operator": "==", "value": value})

            return {"filters": {"operator": "AND", "conditions": filters}}

    documents = [
        Document(
            content="some publication about Alzheimer prevention research done over 2023 patients study",
            meta={"year": 2022, "disease": "Alzheimer", "author": "Michael Butter"},
            id="doc1",
        ),
        Document(
            content="some text about investigation and treatment of Alzheimer disease",
            meta={"year": 2023, "disease": "Alzheimer", "author": "John Bread"},
            id="doc2",
        ),
        Document(
            content="A study on the effectiveness of new therapies for Parkinson's disease",
            meta={"year": 2022, "disease": "Parkinson", "author": "Alice Smith"},
            id="doc3",
        ),
        Document(
            content="An overview of the latest research on the genetics of Parkinson's disease and its implications for treatment",
            meta={"year": 2023, "disease": "Parkinson", "author": "David Jones"},
            id="doc4",
        ),
    ]
    document_store = InMemoryDocumentStore(bm25_algorithm="BM25Plus")
    document_store.write_documents(documents=documents, policy=DuplicatePolicy.OVERWRITE)

    pipeline = Pipeline()
    pipeline.add_component(instance=PromptBuilder('{"disease": "Alzheimer", "year": 2023}'), name="builder")
    pipeline.add_component(instance=QueryMetadataExtractor(), name="metadata_extractor")
    pipeline.add_component(instance=InMemoryBM25Retriever(document_store=document_store), name="retriever")
    pipeline.add_component(instance=DocumentJoiner(), name="document_joiner")

    pipeline.connect("builder.prompt", "metadata_extractor.prompt")
    pipeline.connect("metadata_extractor.filters", "retriever.filters")
    pipeline.connect("retriever.documents", "document_joiner.documents")

    query = "publications 2023 Alzheimer's disease"

    return (
        pipeline,
        [
            PipelineRunData(
                inputs={"retriever": {"query": query}},
                expected_outputs={
                    "document_joiner": {
                        "documents": [
                            Document(
                                content="some text about investigation and treatment of Alzheimer disease",
                                meta={"year": 2023, "disease": "Alzheimer", "author": "John Bread"},
                                id="doc2",
                                score=3.324112496100923,
                            )
                        ]
                    }
                },
                expected_run_order=["builder", "metadata_extractor", "retriever", "document_joiner"],
            )
        ],
    )


@given("a pipeline that has a cycle that would get it stuck", target_fixture="pipeline_data")
def that_has_a_cycle_that_would_get_it_stuck():
    template = """
    You are an experienced and accurate Turkish CX speacialist that classifies customer comments into pre-defined categories below:\n
    Negative experience labels:
    - Late delivery
    - Rotten/spoilt item
    - Bad Courier behavior

    Positive experience labels:
    - Good courier behavior
    - Thanks & appreciation
    - Love message to courier
    - Fast delivery
    - Quality of products

    Create a JSON object as a response. The fields are: 'positive_experience', 'negative_experience'.
    Assign at least one of the pre-defined labels to the given customer comment under positive and negative experience fields.
    If the comment has a positive experience, list the label under 'positive_experience' field.
    If the comments has a negative_experience, list it under the 'negative_experience' field.
    Here is the comment:\n{{ comment }}\n. Just return the category names in the list. If there aren't any, return an empty list.

    {% if invalid_replies and error_message %}
    You already created the following output in a previous attempt: {{ invalid_replies }}
    However, this doesn't comply with the format requirements from above and triggered this Python exception: {{ error_message }}
    Correct the output and try again. Just return the corrected output without any extra explanations.
    {% endif %}
    """
    prompt_builder = PromptBuilder(
        template=template, required_variables=["comment", "invalid_replies", "error_message"]
    )

    @component
    class FakeOutputValidator:
        @component.output_types(
            valid_replies=List[str], invalid_replies=Optional[List[str]], error_message=Optional[str]
        )
        def run(self, replies: List[str]):
            if not getattr(self, "called", False):
                self.called = True
                return {"invalid_replies": ["This is an invalid reply"], "error_message": "this is an error message"}
            return {"valid_replies": replies}

    @component
    class FakeGenerator:
        @component.output_types(replies=List[str])
        def run(self, prompt: str):
            return {"replies": ["This is a valid reply"]}

    llm = FakeGenerator()
    validator = FakeOutputValidator()

    pipeline = Pipeline(max_runs_per_component=1)
    pipeline.add_component("prompt_builder", prompt_builder)

    pipeline.add_component("llm", llm)
    pipeline.add_component("output_validator", validator)

    pipeline.connect("prompt_builder.prompt", "llm.prompt")
    pipeline.connect("llm.replies", "output_validator.replies")
    pipeline.connect("output_validator.invalid_replies", "prompt_builder.invalid_replies")

    pipeline.connect("output_validator.error_message", "prompt_builder.error_message")

    comment = "I loved the quality of the meal but the courier was rude"
    return (pipeline, [PipelineRunData(inputs={"prompt_builder": {"comment": comment}})])


@given("a pipeline that has a loop in the middle", target_fixture="pipeline_data")
def that_has_a_loop_in_the_middle():
    @component
    class FakeGenerator:
        @component.output_types(replies=List[str])
        def run(self, prompt: str):
            replies = []
            if getattr(self, "first_run", True):
                self.first_run = False
                replies.append("No answer")
            else:
                replies.append("42")
            return {"replies": replies}

    @component
    class PromptCleaner:
        @component.output_types(clean_prompt=str)
        def run(self, prompt: str):
            return {"clean_prompt": prompt.strip()}

    routes = [
        {
            "condition": "{{ 'No answer' in replies }}",
            "output": "{{ replies }}",
            "output_name": "invalid_replies",
            "output_type": List[str],
        },
        {
            "condition": "{{ 'No answer' not in replies }}",
            "output": "{{ replies }}",
            "output_name": "valid_replies",
            "output_type": List[str],
        },
    ]

    pipeline = Pipeline(max_runs_per_component=20)
    pipeline.add_component("prompt_cleaner", PromptCleaner())
    pipeline.add_component("prompt_builder", PromptBuilder(template="", variables=["question", "invalid_replies"]))
    pipeline.add_component("llm", FakeGenerator())
    pipeline.add_component("answer_validator", ConditionalRouter(routes=routes))
    pipeline.add_component("answer_builder", AnswerBuilder())

    pipeline.connect("prompt_cleaner.clean_prompt", "prompt_builder.template")
    pipeline.connect("prompt_builder.prompt", "llm.prompt")
    pipeline.connect("llm.replies", "answer_validator.replies")
    pipeline.connect("answer_validator.invalid_replies", "prompt_builder.invalid_replies")
    pipeline.connect("answer_validator.valid_replies", "answer_builder.replies")

    question = "What is the answer?"
    return (
        pipeline,
        [
            PipelineRunData(
                inputs={
                    "prompt_cleaner": {"prompt": "Random template"},
                    "prompt_builder": {"question": question},
                    "answer_builder": {"query": question},
                },
                expected_outputs={
                    "answer_builder": {"answers": [GeneratedAnswer(data="42", query=question, documents=[])]}
                },
                expected_run_order=[
                    "prompt_cleaner",
                    "prompt_builder",
                    "llm",
                    "answer_validator",
                    "prompt_builder",
                    "llm",
                    "answer_validator",
                    "answer_builder",
                ],
            )
        ],
    )


@given("a pipeline that has variadic component that receives a conditional input", target_fixture="pipeline_data")
def that_has_variadic_component_that_receives_a_conditional_input():
    pipe = Pipeline(max_runs_per_component=1)
    routes = [
        {
            "condition": "{{ documents|length > 1 }}",
            "output": "{{ documents }}",
            "output_name": "long",
            "output_type": List[Document],
        },
        {
            "condition": "{{ documents|length <= 1 }}",
            "output": "{{ documents }}",
            "output_name": "short",
            "output_type": List[Document],
        },
    ]

    @component
    class NoOp:
        @component.output_types(documents=List[Document])
        def run(self, documents: List[Document]):
            return {"documents": documents}

    @component
    class CommaSplitter:
        @component.output_types(documents=List[Document])
        def run(self, documents: List[Document]):
            res = []
            current_id = 0
            for doc in documents:
                for split in doc.content.split(","):
                    res.append(Document(content=split, id=str(current_id)))
                    current_id += 1
            return {"documents": res}

    pipe.add_component("conditional_router", ConditionalRouter(routes, unsafe=True))
    pipe.add_component(
        "empty_lines_cleaner", DocumentCleaner(remove_empty_lines=True, remove_extra_whitespaces=False, keep_id=True)
    )
    pipe.add_component("comma_splitter", CommaSplitter())
    pipe.add_component("document_cleaner", DocumentCleaner(keep_id=True))
    pipe.add_component("document_joiner", DocumentJoiner())

    pipe.add_component("noop2", NoOp())
    pipe.add_component("noop3", NoOp())

    pipe.connect("noop2", "noop3")
    pipe.connect("noop3", "conditional_router")

    pipe.connect("conditional_router.long", "empty_lines_cleaner")
    pipe.connect("empty_lines_cleaner", "document_joiner")

    pipe.connect("comma_splitter", "document_cleaner")
    pipe.connect("document_cleaner", "document_joiner")
    pipe.connect("comma_splitter", "document_joiner")

    document = Document(
        id="1000", content="This document has so many, sentences. Like this one, or this one. Or even this other one."
    )

    return pipe, [
        PipelineRunData(
            inputs={"noop2": {"documents": [document]}, "comma_splitter": {"documents": [document]}},
            expected_outputs={
                "conditional_router": {
                    "short": [
                        Document(
                            id="1000",
                            content="This document has so many, sentences. Like this one, or this one. Or even this other one.",
                        )
                    ]
                },
                "document_joiner": {
                    "documents": [
                        Document(id="0", content="This document has so many"),
                        Document(id="1", content=" sentences. Like this one"),
                        Document(id="2", content=" or this one. Or even this other one."),
                    ]
                },
            },
            expected_run_order=[
                "comma_splitter",
                "noop2",
                "document_cleaner",
                "noop3",
                "conditional_router",
                "document_joiner",
            ],
        ),
        PipelineRunData(
            inputs={
                "noop2": {"documents": [document, document]},
                "comma_splitter": {"documents": [document, document]},
            },
            expected_outputs={
                "document_joiner": {
                    "documents": [
                        Document(id="0", content="This document has so many"),
                        Document(id="1", content=" sentences. Like this one"),
                        Document(id="2", content=" or this one. Or even this other one."),
                        Document(id="3", content="This document has so many"),
                        Document(id="4", content=" sentences. Like this one"),
                        Document(id="5", content=" or this one. Or even this other one."),
                        Document(
                            id="1000",
                            content="This document has so many, sentences. Like this one, or this one. Or even this other one.",
                        ),
                    ]
                }
            },
            expected_run_order=[
                "comma_splitter",
                "noop2",
                "document_cleaner",
                "noop3",
                "conditional_router",
                "empty_lines_cleaner",
                "document_joiner",
            ],
        ),
    ]


@given("a pipeline that has a string variadic component", target_fixture="pipeline_data")
def that_has_a_string_variadic_component():
    string_1 = "What's Natural Language Processing?"
    string_2 = "What's is life?"

    pipeline = Pipeline()
    pipeline.add_component("prompt_builder_1", PromptBuilder("Builder 1: {{query}}"))
    pipeline.add_component("prompt_builder_2", PromptBuilder("Builder 2: {{query}}"))
    pipeline.add_component("string_joiner", StringJoiner())

    pipeline.connect("prompt_builder_1.prompt", "string_joiner.strings")
    pipeline.connect("prompt_builder_2.prompt", "string_joiner.strings")

    return (
        pipeline,
        [
            PipelineRunData(
                inputs={"prompt_builder_1": {"query": string_1}, "prompt_builder_2": {"query": string_2}},
                expected_outputs={
                    "string_joiner": {
                        "strings": ["Builder 1: What's Natural Language Processing?", "Builder 2: What's is life?"]
                    }
                },
                expected_run_order=["prompt_builder_1", "prompt_builder_2", "string_joiner"],
            )
        ],
    )
