from typing import List

from pytest_bdd import scenarios, given
import pytest

from haystack import Pipeline, Document, component
from haystack.dataclasses import ChatMessage
from haystack.components.builders import PromptBuilder
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.others import Multiplexer
from haystack.core.errors import PipelineMaxLoops, PipelineRuntimeError
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


pytestmark = pytest.mark.integration

scenarios("pipeline_run.feature")


@given("a pipeline that has no components", target_fixture="pipeline_data")
def pipeline_that_has_no_components():
    pipeline = Pipeline()
    inputs = {}
    expected_outputs = {}
    return pipeline, inputs, expected_outputs, []


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
        {"first_addition": {"value": 1}},
        {"second_addition": {"result": 7}},
        ["first_addition", "double", "second_addition"],
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
    return pipe, {"first": {"x": 1}}, PipelineMaxLoops


@given("a pipeline that is really complex with lots of components, forks, and loops", target_fixture="pipeline_data")
def pipeline_complex():
    pipeline = Pipeline(max_loops_allowed=2)
    pipeline.add_component("greet_first", Greet(message="Hello, the value is {value}."))
    pipeline.add_component("accumulate_1", Accumulate())
    pipeline.add_component("add_two", AddFixedValue(add=2))
    pipeline.add_component("parity", Parity())
    pipeline.add_component("add_one", AddFixedValue(add=1))
    pipeline.add_component("accumulate_2", Accumulate())

    pipeline.add_component("multiplexer", Multiplexer(type_=int))
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
    pipeline.connect("add_one", "multiplexer.value")
    pipeline.connect("multiplexer", "below_10")

    pipeline.connect("below_10.below", "double")
    pipeline.connect("double", "multiplexer.value")

    pipeline.connect("below_10.above", "accumulate_2")
    pipeline.connect("accumulate_2", "diff.second_value")

    pipeline.connect("greet_enumerator", "enumerate")
    pipeline.connect("enumerate.second", "sum.values")

    pipeline.connect("enumerate.first", "add_three.value")
    pipeline.connect("add_three", "sum.values")

    return (
        pipeline,
        {"greet_first": {"value": 1}, "greet_enumerator": {"value": 1}},
        {"accumulate_3": {"value": -7}, "add_five": {"result": -6}},
        [
            "greet_first",
            "accumulate_1",
            "add_two",
            "parity",
            "add_one",
            "multiplexer",
            "below_10",
            "double",
            "multiplexer",
            "below_10",
            "double",
            "multiplexer",
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
        [{"with_defaults": {"a": 40, "b": 30}}, {"with_defaults": {"a": 40}}],
        [{"with_defaults": {"c": 70}}, {"with_defaults": {"c": 42}}],
        [["with_defaults"], ["with_defaults"]],
    )


@given("a pipeline that has two loops of identical lengths", target_fixture="pipeline_data")
def pipeline_that_has_two_loops_of_identical_lengths():
    pipeline = Pipeline(max_loops_allowed=10)
    pipeline.add_component("multiplexer", Multiplexer(type_=int))
    pipeline.add_component("remainder", Remainder(divisor=3))
    pipeline.add_component("add_one", AddFixedValue(add=1))
    pipeline.add_component("add_two", AddFixedValue(add=2))

    pipeline.connect("multiplexer.value", "remainder.value")
    pipeline.connect("remainder.remainder_is_1", "add_two.value")
    pipeline.connect("remainder.remainder_is_2", "add_one.value")
    pipeline.connect("add_two", "multiplexer.value")
    pipeline.connect("add_one", "multiplexer.value")

    return (
        pipeline,
        [
            {"multiplexer": {"value": 0}},
            {"multiplexer": {"value": 3}},
            {"multiplexer": {"value": 4}},
            {"multiplexer": {"value": 5}},
            {"multiplexer": {"value": 6}},
        ],
        [
            {"remainder": {"remainder_is_0": 0}},
            {"remainder": {"remainder_is_0": 3}},
            {"remainder": {"remainder_is_0": 6}},
            {"remainder": {"remainder_is_0": 6}},
            {"remainder": {"remainder_is_0": 6}},
        ],
        [
            ["multiplexer", "remainder"],
            ["multiplexer", "remainder"],
            ["multiplexer", "remainder", "add_two", "multiplexer", "remainder"],
            ["multiplexer", "remainder", "add_one", "multiplexer", "remainder"],
            ["multiplexer", "remainder"],
        ],
    )


@given("a pipeline that has two loops of different lengths", target_fixture="pipeline_data")
def pipeline_that_has_two_loops_of_different_lengths():
    pipeline = Pipeline(max_loops_allowed=10)
    pipeline.add_component("multiplexer", Multiplexer(type_=int))
    pipeline.add_component("remainder", Remainder(divisor=3))
    pipeline.add_component("add_one", AddFixedValue(add=1))
    pipeline.add_component("add_two_1", AddFixedValue(add=1))
    pipeline.add_component("add_two_2", AddFixedValue(add=1))

    pipeline.connect("multiplexer.value", "remainder.value")
    pipeline.connect("remainder.remainder_is_1", "add_two_1.value")
    pipeline.connect("add_two_1", "add_two_2.value")
    pipeline.connect("add_two_2", "multiplexer")
    pipeline.connect("remainder.remainder_is_2", "add_one.value")
    pipeline.connect("add_one", "multiplexer")

    return (
        pipeline,
        [
            {"multiplexer": {"value": 0}},
            {"multiplexer": {"value": 3}},
            {"multiplexer": {"value": 4}},
            {"multiplexer": {"value": 5}},
            {"multiplexer": {"value": 6}},
        ],
        [
            {"remainder": {"remainder_is_0": 0}},
            {"remainder": {"remainder_is_0": 3}},
            {"remainder": {"remainder_is_0": 6}},
            {"remainder": {"remainder_is_0": 6}},
            {"remainder": {"remainder_is_0": 6}},
        ],
        [
            ["multiplexer", "remainder"],
            ["multiplexer", "remainder"],
            ["multiplexer", "remainder", "add_two_1", "add_two_2", "multiplexer", "remainder"],
            ["multiplexer", "remainder", "add_one", "multiplexer", "remainder"],
            ["multiplexer", "remainder"],
        ],
    )


@given("a pipeline that has a single loop with two conditional branches", target_fixture="pipeline_data")
def pipeline_that_has_a_single_loop_with_two_conditional_branches():
    accumulator = Accumulate()
    pipeline = Pipeline(max_loops_allowed=10)

    pipeline.add_component("add_one", AddFixedValue(add=1))
    pipeline.add_component("multiplexer", Multiplexer(type_=int))
    pipeline.add_component("below_10", Threshold(threshold=10))
    pipeline.add_component("below_5", Threshold(threshold=5))
    pipeline.add_component("add_three", AddFixedValue(add=3))
    pipeline.add_component("accumulator", accumulator)
    pipeline.add_component("add_two", AddFixedValue(add=2))

    pipeline.connect("add_one.result", "multiplexer")
    pipeline.connect("multiplexer.value", "below_10.value")
    pipeline.connect("below_10.below", "accumulator.value")
    pipeline.connect("accumulator.value", "below_5.value")
    pipeline.connect("below_5.above", "add_three.value")
    pipeline.connect("below_5.below", "multiplexer")
    pipeline.connect("add_three.result", "multiplexer")
    pipeline.connect("below_10.above", "add_two.value")

    return (
        pipeline,
        {"add_one": {"value": 3}},
        {"add_two": {"result": 13}},
        [
            "add_one",
            "multiplexer",
            "below_10",
            "accumulator",
            "below_5",
            "multiplexer",
            "below_10",
            "accumulator",
            "below_5",
            "add_three",
            "multiplexer",
            "below_10",
            "add_two",
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
        [{"hello": {"word": "Alice"}}, {"hello": {"word": "Alice"}, "fstring": {"template": "Received: {greeting}"}}],
        [
            {"splitter": {"output": ["This", "is", "the", "greeting:", "Hello,", "Alice!!"]}},
            {"splitter": {"output": ["Received:", "Hello,", "Alice!"]}},
        ],
        [["hello", "fstring", "splitter"], ["hello", "fstring", "splitter"]],
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
        [{"add_one": {"value": 1}}, {"add_one": {"value": 2}}],
        [{"add_three": {"result": 15}}, {"double": {"value": 6}}],
        [["add_one", "parity", "add_ten", "add_three"], ["add_one", "parity", "double"]],
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
        {"add_one": {"value": 1}},
        {"add_one_again": {"result": 6}, "add_ten": {"result": 12}, "double": {"value": 4}},
        ["add_one", "repeat", "double", "add_ten", "add_three", "add_one_again"],
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
        {"first_addition": {"value": 1}, "third_addition": {"value": 1}},
        {"fourth_addition": {"result": 3}},
        ["first_addition", "second_addition", "third_addition", "diff", "fourth_addition"],
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
            {"add_one": {"value": 1}, "add_two": {"add": 2}, "add_two_as_well": {"add": 2}},
            {"add_one": {"value": 2}, "add_two": {"add": 2}, "add_two_as_well": {"add": 2}},
        ],
        [{"add_two": {"result": 8}, "add_two_as_well": {"result": 8}}, {"diff": {"difference": 7}}],
        [
            ["add_one", "parity", "add_four", "add_two", "add_two_as_well"],
            ["add_one", "parity", "double", "add_ten", "diff"],
        ],
    )


@given("a pipeline that has two branches, one of which loops back", target_fixture="pipeline_data")
def pipeline_that_has_two_branches_one_of_which_loops_back():
    pipeline = Pipeline(max_loops_allowed=10)
    pipeline.add_component("add_zero", AddFixedValue(add=0))
    pipeline.add_component("multiplexer", Multiplexer(type_=int))
    pipeline.add_component("sum", Sum())
    pipeline.add_component("below_10", Threshold(threshold=10))
    pipeline.add_component("add_one", AddFixedValue(add=1))
    pipeline.add_component("counter", Accumulate())
    pipeline.add_component("add_two", AddFixedValue(add=2))

    pipeline.connect("add_zero", "multiplexer.value")
    pipeline.connect("multiplexer", "below_10.value")
    pipeline.connect("below_10.below", "add_one.value")
    pipeline.connect("add_one.result", "counter.value")
    pipeline.connect("counter.value", "multiplexer.value")
    pipeline.connect("below_10.above", "add_two.value")
    pipeline.connect("add_two.result", "sum.values")

    return (
        pipeline,
        {"add_zero": {"value": 8}, "sum": {"values": 2}},
        {"sum": {"total": 23}},
        [
            "add_zero",
            "multiplexer",
            "below_10",
            "add_one",
            "counter",
            "multiplexer",
            "below_10",
            "add_one",
            "counter",
            "multiplexer",
            "below_10",
            "add_two",
            "sum",
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
        {"mangler1": {"input_list": input_list}, "mangler2": {"input_list": input_list}},
        {"concat1": {"output": ["foo", "bar", "extra_item"]}, "concat2": {"output": ["foo", "bar", "extra_item"]}},
        ["mangler1", "mangler2", "concat1", "concat2"],
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
        {"prompt_builder": {"prompt_source": messages}, "mm1": params, "mm2": params},
        {
            "mm1": {
                "merged_message": "Always respond in English even if some input data is in other languages.\nTell me about Berlin"
            },
            "mm2": {"merged_message": "Fake message"},
        },
        ["prompt_builder", "mm1", "llm", "mm2"],
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
    pipeline.add_component("multiplexer", Multiplexer(List[Document]))

    pipeline.connect("retriever", "multiplexer")
    pipeline.connect("multiplexer", "prompt_builder.documents")
    return (
        pipeline,
        {"query": "This is my question"},
        {
            "prompt_builder": {
                "prompt": "Given this documents: This is a simple document Answer this question: This is my question"
            }
        },
        ["retriever", "multiplexer", "prompt_builder"],
    )


@given("a pipeline that has a component that doesn't return a dictionary", target_fixture="pipeline_data")
def pipeline_that_has_a_component_that_doesnt_return_a_dictionary():
    BrokenComponent = component_class(
        "BrokenComponent", input_types={"a": int}, output_types={"b": int}, output=1  # type:ignore
    )

    pipe = Pipeline(max_loops_allowed=10)
    pipe.add_component("comp", BrokenComponent())
    return pipe, {"comp": {"a": 1}}, PipelineRuntimeError
