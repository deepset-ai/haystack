from typing import List, Optional, Dict, Any

from pytest_bdd import scenarios, given
import pytest

from haystack import Pipeline, Document, component
from haystack.dataclasses import ChatMessage, GeneratedAnswer
from haystack.components.routers import ConditionalRouter
from haystack.components.builders import PromptBuilder, AnswerBuilder
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.joiners import BranchJoiner, DocumentJoiner
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
    SelfLoop,
)
from haystack.testing.factory import component_class

from test.core.pipeline.features.conftest import PipelineRunData

pytestmark = pytest.mark.integration

scenarios("pipeline_run.feature")


@given("a pipeline that has no components", target_fixture="pipeline_data")
def pipeline_that_has_no_components():
    pipeline = Pipeline()
    inputs = {}
    expected_outputs = {}
    return pipeline, [PipelineRunData(inputs=inputs, expected_outputs=expected_outputs)]


@given("a pipeline that is linear", target_fixture="pipeline_data")
def pipeline_that_is_linear():
    pipeline = Pipeline()
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
    def custom_init(self):
        component.set_input_type(self, "x", int)
        component.set_input_type(self, "y", int, 1)
        component.set_output_types(self, a=int, b=int)

    FakeComponent = component_class("FakeComponent", output={"a": 1, "b": 1}, extra_fields={"__init__": custom_init})
    pipe = Pipeline(max_loops_allowed=1)
    pipe.add_component("first", FakeComponent())
    pipe.add_component("second", FakeComponent())
    pipe.connect("first.a", "second.x")
    pipe.connect("second.b", "first.y")
    return pipe, [PipelineRunData({"first": {"x": 1}})]


@given("a pipeline that is really complex with lots of components, forks, and loops", target_fixture="pipeline_data")
def pipeline_complex():
    pipeline = Pipeline(max_loops_allowed=2)
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
                    "accumulate_1",
                    "add_two",
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
                    "greet_enumerator",
                    "enumerate",
                    "add_three",
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

    pipeline = Pipeline()
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
    pipeline = Pipeline(max_loops_allowed=10)
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
    pipeline = Pipeline(max_loops_allowed=10)
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
    pipeline = Pipeline(max_loops_allowed=10)

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
    pipeline = Pipeline()
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
    pipeline = Pipeline()
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
    pipeline = Pipeline()
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
    pipeline = Pipeline()
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
    pipeline = Pipeline()
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
    pipeline = Pipeline(max_loops_allowed=10)
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

    pipe = Pipeline()
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
            return {"merged_message": "\n".join(t.content for t in messages)}

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

    pipe = Pipeline()
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

    pipeline = Pipeline()
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

    pipe = Pipeline(max_loops_allowed=10)
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

    pipe = Pipeline()

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

    pipe = Pipeline()

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


@given("a pipeline that has a component with only default inputs as first to run", target_fixture="pipeline_data")
def pipeline_that_has_a_component_with_only_default_inputs_as_first_to_run():
    """
    This tests verifies that a Pipeline doesn't get stuck running in a loop if
    it has all the following characterics:
    - The first Component has all defaults for its inputs
    - The first Component receives one input from the user
    - The first Component receives one input from a loop in the Pipeline
    - The second Component has at least one default input
    """

    def fake_generator_run(self, prompt: str, generation_kwargs: Optional[Dict[str, Any]] = None):
        # Simple hack to simulate a model returning a different reply after the
        # the first time it's called
        if getattr(fake_generator_run, "called", False):
            return {"replies": ["Rome"]}
        fake_generator_run.called = True
        return {"replies": ["Paris"]}

    FakeGenerator = component_class(
        "FakeGenerator",
        input_types={"prompt": str, "generation_kwargs": Optional[Dict[str, Any]]},
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

    pipe = Pipeline()

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
    "a pipeline that has only a single component that sends one of its outputs to itself",
    target_fixture="pipeline_data",
)
def pipeline_that_has_a_single_component_that_send_one_of_outputs_to_itself():
    pipeline = Pipeline(max_loops_allowed=10)
    pipeline.add_component("self_loop", SelfLoop())
    pipeline.connect("self_loop.current_value", "self_loop.values")

    return (
        pipeline,
        [
            PipelineRunData(
                inputs={"self_loop": {"values": 5}},
                expected_outputs={"self_loop": {"final_result": 0}},
                expected_run_order=["self_loop", "self_loop", "self_loop", "self_loop", "self_loop"],
            )
        ],
    )


@given("a pipeline that has a component that sends one of its outputs to itself", target_fixture="pipeline_data")
def pipeline_that_has_a_component_that_sends_one_of_its_outputs_to_itself():
    pipeline = Pipeline(max_loops_allowed=10)
    pipeline.add_component("add_1", AddFixedValue())
    pipeline.add_component("self_loop", SelfLoop())
    pipeline.add_component("add_2", AddFixedValue())
    pipeline.connect("add_1", "self_loop.values")
    pipeline.connect("self_loop.current_value", "self_loop.values")
    pipeline.connect("self_loop.final_result", "add_2.value")

    return (
        pipeline,
        [
            PipelineRunData(
                inputs={"add_1": {"value": 5}},
                expected_outputs={"add_2": {"result": 1}},
                expected_run_order=[
                    "add_1",
                    "self_loop",
                    "self_loop",
                    "self_loop",
                    "self_loop",
                    "self_loop",
                    "self_loop",
                    "add_2",
                ],
            )
        ],
    )


@given(
    "a pipeline that has multiple branches that merge into a component with a single variadic input",
    target_fixture="pipeline_data",
)
def pipeline_that_has_multiple_branches_that_merge_into_a_component_with_a_single_variadic_input():
    pipeline = Pipeline()
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
    pipeline = Pipeline()
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
    pipeline = Pipeline()
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
    pipeline = Pipeline(max_loops_allowed=10)
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

    pipeline = Pipeline()
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

    pipeline = Pipeline()
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

    pipeline = Pipeline()
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

    pipeline = Pipeline()
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

    pipeline = Pipeline()
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
