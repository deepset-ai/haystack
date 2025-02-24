import json
from copy import deepcopy
from typing import List, Optional, Dict, Any
import re

from pytest_bdd import scenarios, given
from unittest.mock import ANY
import pytest
import pandas as pd

from haystack import Document, component
from haystack.document_stores.types import DuplicatePolicy
from haystack.dataclasses import ChatMessage, GeneratedAnswer, TextContent, ByteStream
from haystack.components.routers import ConditionalRouter, FileTypeRouter
from haystack.components.builders import PromptBuilder, AnswerBuilder, ChatPromptBuilder
from haystack.components.converters import (
    OutputAdapter,
    JSONConverter,
    TextFileToDocument,
    CSVToDocument,
    HTMLToDocument,
)
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.joiners import BranchJoiner, DocumentJoiner, AnswerJoiner, StringJoiner
from haystack.core.component.types import Variadic
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

pytestmark = [pytest.mark.usefixtures("pipeline_class"), pytest.mark.integration]

scenarios("pipeline_run.feature")


@given("a pipeline that has no components", target_fixture="pipeline_data")
def pipeline_that_has_no_components(pipeline_class):
    pipeline = pipeline_class(max_runs_per_component=1)
    inputs = {}
    expected_outputs = {}
    return pipeline, [PipelineRunData(inputs=inputs, expected_outputs=expected_outputs)]


@given("a pipeline that is linear", target_fixture="pipeline_data")
def pipeline_that_is_linear(pipeline_class):
    pipeline = pipeline_class(max_runs_per_component=1)
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
                expected_component_calls={
                    ("first_addition", 1): {"value": 1, "add": None},
                    ("double", 1): {"value": 3},
                    ("second_addition", 1): {"value": 6, "add": None},
                },
            )
        ],
    )


@given("a pipeline that has an infinite loop", target_fixture="pipeline_data")
def pipeline_that_has_an_infinite_loop(pipeline_class):
    routes = [
        {"condition": "{{number > 2}}", "output": "{{number}}", "output_name": "big_number", "output_type": int},
        {"condition": "{{number <= 2}}", "output": "{{number + 2}}", "output_name": "small_number", "output_type": int},
    ]

    main_input = BranchJoiner(int)
    first_router = ConditionalRouter(routes=routes)
    second_router = ConditionalRouter(routes=routes)

    pipe = pipeline_class(max_runs_per_component=1)
    pipe.add_component("main_input", main_input)
    pipe.add_component("first_router", first_router)
    pipe.add_component("second_router", second_router)

    pipe.connect("main_input", "first_router.number")
    pipe.connect("first_router.big_number", "second_router.number")
    pipe.connect("second_router.big_number", "main_input")

    return pipe, [PipelineRunData({"main_input": {"value": 3}})]


@given("a pipeline that is really complex with lots of components, forks, and loops", target_fixture="pipeline_data")
def pipeline_complex(pipeline_class):
    pipeline = pipeline_class(max_runs_per_component=2)
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
                expected_component_calls={
                    ("greet_first", 1): {"value": 1, "log_level": None, "message": None},
                    ("greet_enumerator", 1): {"value": 1, "log_level": None, "message": None},
                    ("accumulate_1", 1): {"value": 1},
                    ("add_two", 1): {"value": 1, "add": None},
                    ("parity", 1): {"value": 3},
                    ("add_one", 1): {"value": 3, "add": None},
                    ("branch_joiner", 1): {"value": [4]},
                    ("below_10", 1): {"value": 4, "threshold": None},
                    ("double", 1): {"value": 4},
                    ("branch_joiner", 2): {"value": [8]},
                    ("below_10", 2): {"value": 8, "threshold": None},
                    ("double", 2): {"value": 8},
                    ("branch_joiner", 3): {"value": [16]},
                    ("below_10", 3): {"value": 16, "threshold": None},
                    ("accumulate_2", 1): {"value": 16},
                    ("enumerate", 1): {"value": 1},
                    ("add_three", 1): {"value": 1, "add": None},
                    ("sum", 1): {"values": [1, 4]},
                    ("diff", 1): {"first_value": 5, "second_value": 16},
                    ("greet_one_last_time", 1): {"value": -11, "log_level": None, "message": None},
                    ("replicate", 1): {"value": -11},
                    ("add_five", 1): {"value": -11, "add": None},
                    ("add_four", 1): {"value": -11, "add": None},
                    ("accumulate_3", 1): {"value": -7},
                },
            )
        ],
    )


@given("a pipeline that has a single component with a default input", target_fixture="pipeline_data")
def pipeline_that_has_a_single_component_with_a_default_input(pipeline_class):
    @component
    class WithDefault:
        @component.output_types(b=int)
        def run(self, a: int, b: int = 2):
            return {"c": a + b}

    pipeline = pipeline_class(max_runs_per_component=1)
    pipeline.add_component("with_defaults", WithDefault())

    return (
        pipeline,
        [
            PipelineRunData(
                inputs={"with_defaults": {"a": 40, "b": 30}},
                expected_outputs={"with_defaults": {"c": 70}},
                expected_component_calls={("with_defaults", 1): {"a": 40, "b": 30}},
            ),
            PipelineRunData(
                inputs={"with_defaults": {"a": 40}},
                expected_outputs={"with_defaults": {"c": 42}},
                expected_component_calls={("with_defaults", 1): {"a": 40, "b": 2}},
            ),
        ],
    )


@given("a pipeline that has two loops of identical lengths", target_fixture="pipeline_data")
def pipeline_that_has_two_loops_of_identical_lengths(pipeline_class):
    pipeline = pipeline_class(max_runs_per_component=10)
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
                expected_component_calls={("branch_joiner", 1): {"value": [0]}, ("remainder", 1): {"value": 0}},
            ),
            PipelineRunData(
                inputs={"branch_joiner": {"value": 3}},
                expected_outputs={"remainder": {"remainder_is_0": 3}},
                expected_component_calls={("branch_joiner", 1): {"value": [3]}, ("remainder", 1): {"value": 3}},
            ),
            PipelineRunData(
                inputs={"branch_joiner": {"value": 4}},
                expected_outputs={"remainder": {"remainder_is_0": 6}},
                expected_component_calls={
                    ("branch_joiner", 1): {"value": [4]},
                    ("remainder", 1): {"value": 4},
                    ("add_two", 1): {"value": 4, "add": None},
                    ("branch_joiner", 2): {"value": [6]},
                    ("remainder", 2): {"value": 6},
                },
            ),
            PipelineRunData(
                inputs={"branch_joiner": {"value": 5}},
                expected_outputs={"remainder": {"remainder_is_0": 6}},
                expected_component_calls={
                    ("branch_joiner", 1): {"value": [5]},
                    ("remainder", 1): {"value": 5},
                    ("add_one", 1): {"value": 5, "add": None},
                    ("branch_joiner", 2): {"value": [6]},
                    ("remainder", 2): {"value": 6},
                },
            ),
            PipelineRunData(
                inputs={"branch_joiner": {"value": 6}},
                expected_outputs={"remainder": {"remainder_is_0": 6}},
                expected_component_calls={("branch_joiner", 1): {"value": [6]}, ("remainder", 1): {"value": 6}},
            ),
        ],
    )


@given("a pipeline that has two loops of different lengths", target_fixture="pipeline_data")
def pipeline_that_has_two_loops_of_different_lengths(pipeline_class):
    pipeline = pipeline_class(max_runs_per_component=10)
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
                expected_component_calls={("branch_joiner", 1): {"value": [0]}, ("remainder", 1): {"value": 0}},
            ),
            PipelineRunData(
                inputs={"branch_joiner": {"value": 3}},
                expected_outputs={"remainder": {"remainder_is_0": 3}},
                expected_component_calls={("branch_joiner", 1): {"value": [3]}, ("remainder", 1): {"value": 3}},
            ),
            PipelineRunData(
                inputs={"branch_joiner": {"value": 4}},
                expected_outputs={"remainder": {"remainder_is_0": 6}},
                expected_component_calls={
                    ("branch_joiner", 1): {"value": [4]},
                    ("remainder", 1): {"value": 4},
                    ("add_two_1", 1): {"value": 4, "add": None},
                    ("add_two_2", 1): {"value": 5, "add": None},
                    ("branch_joiner", 2): {"value": [6]},
                    ("remainder", 2): {"value": 6},
                },
            ),
            PipelineRunData(
                inputs={"branch_joiner": {"value": 5}},
                expected_outputs={"remainder": {"remainder_is_0": 6}},
                expected_component_calls={
                    ("branch_joiner", 1): {"value": [5]},
                    ("remainder", 1): {"value": 5},
                    ("add_one", 1): {"value": 5, "add": None},
                    ("branch_joiner", 2): {"value": [6]},
                    ("remainder", 2): {"value": 6},
                },
            ),
            PipelineRunData(
                inputs={"branch_joiner": {"value": 6}},
                expected_outputs={"remainder": {"remainder_is_0": 6}},
                expected_component_calls={("branch_joiner", 1): {"value": [6]}, ("remainder", 1): {"value": 6}},
            ),
        ],
    )


@given("a pipeline that has a single loop with two conditional branches", target_fixture="pipeline_data")
def pipeline_that_has_a_single_loop_with_two_conditional_branches(pipeline_class):
    accumulator = Accumulate()
    pipeline = pipeline_class(max_runs_per_component=10)

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
                expected_component_calls={
                    ("accumulator", 1): {"value": 4},
                    ("accumulator", 2): {"value": 4},
                    ("add_one", 1): {"add": None, "value": 3},
                    ("add_three", 1): {"add": None, "value": 8},
                    ("add_two", 1): {"add": None, "value": 11},
                    ("below_10", 1): {"threshold": None, "value": 4},
                    ("below_10", 2): {"threshold": None, "value": 4},
                    ("below_10", 3): {"threshold": None, "value": 11},
                    ("below_5", 1): {"threshold": None, "value": 4},
                    ("below_5", 2): {"threshold": None, "value": 8},
                    ("branch_joiner", 1): {"value": [4]},
                    ("branch_joiner", 2): {"value": [4]},
                    ("branch_joiner", 3): {"value": [11]},
                },
            )
        ],
    )


@given("a pipeline that has a component with dynamic inputs defined in init", target_fixture="pipeline_data")
def pipeline_that_has_a_component_with_dynamic_inputs_defined_in_init(pipeline_class):
    pipeline = pipeline_class(max_runs_per_component=1)
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
                expected_component_calls={
                    ("fstring", 1): {"greeting": "Hello, Alice!", "template": None},
                    ("hello", 1): {"word": "Alice"},
                    ("splitter", 1): {"sentence": "This is the greeting: Hello, Alice!!"},
                },
            ),
            PipelineRunData(
                inputs={"hello": {"word": "Alice"}, "fstring": {"template": "Received: {greeting}"}},
                expected_outputs={"splitter": {"output": ["Received:", "Hello,", "Alice!"]}},
                expected_component_calls={
                    ("fstring", 1): {"greeting": "Hello, Alice!", "template": "Received: {greeting}"},
                    ("hello", 1): {"word": "Alice"},
                    ("splitter", 1): {"sentence": "Received: Hello, Alice!"},
                },
            ),
        ],
    )


@given("a pipeline that has two branches that don't merge", target_fixture="pipeline_data")
def pipeline_that_has_two_branches_that_dont_merge(pipeline_class):
    pipeline = pipeline_class(max_runs_per_component=1)
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
                expected_component_calls={
                    ("add_one", 1): {"add": None, "value": 1},
                    ("add_ten", 1): {"add": None, "value": 2},
                    ("add_three", 1): {"add": None, "value": 12},
                    ("parity", 1): {"value": 2},
                },
            ),
            PipelineRunData(
                inputs={"add_one": {"value": 2}},
                expected_outputs={"double": {"value": 6}},
                expected_component_calls={
                    ("add_one", 1): {"add": None, "value": 2},
                    ("double", 1): {"value": 3},
                    ("parity", 1): {"value": 3},
                },
            ),
        ],
    )


@given("a pipeline that has three branches that don't merge", target_fixture="pipeline_data")
def pipeline_that_has_three_branches_that_dont_merge(pipeline_class):
    pipeline = pipeline_class(max_runs_per_component=1)
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
                expected_component_calls={
                    ("add_one", 1): {"add": None, "value": 1},
                    ("add_one_again", 1): {"add": None, "value": 5},
                    ("add_ten", 1): {"add": None, "value": 2},
                    ("add_three", 1): {"add": None, "value": 2},
                    ("double", 1): {"value": 2},
                    ("repeat", 1): {"value": 2},
                },
            )
        ],
    )


@given("a pipeline that has two branches that merge", target_fixture="pipeline_data")
def pipeline_that_has_two_branches_that_merge(pipeline_class):
    pipeline = pipeline_class(max_runs_per_component=1)
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
                expected_component_calls={
                    ("diff", 1): {"first_value": 5, "second_value": 3},
                    ("first_addition", 1): {"add": None, "value": 1},
                    ("fourth_addition", 1): {"add": None, "value": 2},
                    ("second_addition", 1): {"add": None, "value": 3},
                    ("third_addition", 1): {"add": None, "value": 1},
                },
            )
        ],
    )


@given(
    "a pipeline that has different combinations of branches that merge and do not merge", target_fixture="pipeline_data"
)
def pipeline_that_has_different_combinations_of_branches_that_merge_and_do_not_merge(pipeline_class):
    pipeline = pipeline_class(max_runs_per_component=1)
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
                expected_component_calls={
                    ("add_four", 1): {"add": None, "value": 2},
                    ("add_one", 1): {"add": None, "value": 1},
                    ("add_two", 1): {"add": 2, "value": 6},
                    ("add_two_as_well", 1): {"add": 2, "value": 6},
                    ("parity", 1): {"value": 2},
                },
            ),
            PipelineRunData(
                inputs={"add_one": {"value": 2}, "add_two": {"add": 2}, "add_two_as_well": {"add": 2}},
                expected_outputs={"diff": {"difference": 7}},
                expected_component_calls={
                    ("add_one", 1): {"add": None, "value": 2},
                    ("add_ten", 1): {"add": None, "value": 3},
                    ("diff", 1): {"first_value": 13, "second_value": 6},
                    ("double", 1): {"value": 3},
                    ("parity", 1): {"value": 3},
                },
            ),
        ],
    )


@given("a pipeline that has two branches, one of which loops back", target_fixture="pipeline_data")
def pipeline_that_has_two_branches_one_of_which_loops_back(pipeline_class):
    pipeline = pipeline_class(max_runs_per_component=10)
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
                expected_component_calls={
                    ("add_one", 1): {"add": None, "value": 8},
                    ("add_one", 2): {"add": None, "value": 9},
                    ("add_two", 1): {"add": None, "value": 19},
                    ("add_zero", 1): {"add": None, "value": 8},
                    ("below_10", 1): {"threshold": None, "value": 8},
                    ("below_10", 2): {"threshold": None, "value": 9},
                    ("below_10", 3): {"threshold": None, "value": 19},
                    ("branch_joiner", 1): {"value": [8]},
                    ("branch_joiner", 2): {"value": [9]},
                    ("branch_joiner", 3): {"value": [19]},
                    ("counter", 1): {"value": 9},
                    ("counter", 2): {"value": 10},
                    ("sum", 1): {"values": [2, 21]},
                },
            )
        ],
    )


@given("a pipeline that has a component with mutable input", target_fixture="pipeline_data")
def pipeline_that_has_a_component_with_mutable_input(pipeline_class):
    @component
    class InputMangler:
        @component.output_types(mangled_list=List[str])
        def run(self, input_list: List[str]):
            input_list.append("extra_item")
            return {"mangled_list": input_list}

    pipe = pipeline_class(max_runs_per_component=1)
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
                expected_component_calls={
                    ("concat1", 1): {"inputs": [["foo", "bar", "extra_item"]]},
                    ("concat2", 1): {"inputs": [["foo", "bar", "extra_item"]]},
                    ("mangler1", 1): {"input_list": ["foo", "bar"]},
                    ("mangler2", 1): {"input_list": ["foo", "bar"]},
                },
            )
        ],
    )


@given("a pipeline that has a component with mutable output sent to multiple inputs", target_fixture="pipeline_data")
def pipeline_that_has_a_component_with_mutable_output_sent_to_multiple_inputs(pipeline_class):
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

    pipe = pipeline_class(max_runs_per_component=1)
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
                expected_component_calls={
                    ("llm", 1): {
                        "messages": [
                            ChatMessage(
                                _role="system",
                                _content=[
                                    TextContent(
                                        text="Always respond in English even if some input data is in other languages."
                                    )
                                ],
                                _name=None,
                                _meta={},
                            ),
                            ChatMessage(
                                _role="user", _content=[TextContent(text="Tell me about Berlin")], _name=None, _meta={}
                            ),
                        ]
                    },
                    ("mm1", 1): {
                        "messages": [
                            ChatMessage(
                                _role="system",
                                _content=[
                                    TextContent(
                                        text="Always respond in English even if some input data is in other languages."
                                    )
                                ],
                                _name=None,
                                _meta={},
                            ),
                            ChatMessage(
                                _role="user", _content=[TextContent(text="Tell me about Berlin")], _name=None, _meta={}
                            ),
                        ],
                        "metadata": {"meta2": "value2", "metadata_key": "metadata_value"},
                    },
                    ("mm2", 1): {
                        "messages": [
                            ChatMessage(
                                _role="assistant", _content=[TextContent(text="Fake message")], _name=None, _meta={}
                            )
                        ],
                        "metadata": {"meta2": "value2", "metadata_key": "metadata_value"},
                    },
                    ("prompt_builder", 1): {
                        "prompt_source": [
                            ChatMessage(
                                _role="system",
                                _content=[
                                    TextContent(
                                        text="Always respond in English even if some input data is in other languages."
                                    )
                                ],
                                _name=None,
                                _meta={},
                            ),
                            ChatMessage(
                                _role="user", _content=[TextContent(text="Tell me about Berlin")], _name=None, _meta={}
                            ),
                        ]
                    },
                },
            )
        ],
    )


@given(
    "a pipeline that has a greedy and variadic component after a component with default input",
    target_fixture="pipeline_data",
)
def pipeline_that_has_a_greedy_and_variadic_component_after_a_component_with_default_input(pipeline_class):
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

    pipeline = pipeline_class(max_runs_per_component=1)
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
                expected_component_calls={
                    ("branch_joiner", 1): {
                        "value": [
                            [
                                Document(
                                    id="328f0cbb6722c5cfa290aa2b78bcda8dc5afa09f0e2c23092afc502ba89c85e7",
                                    content="This is a simple document",
                                    score=0.5993376509412102,
                                )
                            ]
                        ]
                    },
                    ("prompt_builder", 1): {
                        "documents": [
                            Document(
                                id="328f0cbb6722c5cfa290aa2b78bcda8dc5afa09f0e2c23092afc502ba89c85e7",
                                content="This is a simple document",
                                score=0.5993376509412102,
                            )
                        ],
                        "query": "This is my question",
                        "template": None,
                        "template_variables": None,
                    },
                    ("retriever", 1): {
                        "filters": None,
                        "query": "This is my question",
                        "scale_score": None,
                        "top_k": None,
                    },
                },
            )
        ],
    )


@given("a pipeline that has a component that doesn't return a dictionary", target_fixture="pipeline_data")
def pipeline_that_has_a_component_that_doesnt_return_a_dictionary(pipeline_class):
    BrokenComponent = component_class(
        "BrokenComponent",
        input_types={"a": int},
        output_types={"b": int},
        output=1,  # type:ignore
    )

    pipe = pipeline_class(max_runs_per_component=10)
    pipe.add_component("comp", BrokenComponent())
    return pipe, [PipelineRunData({"comp": {"a": 1}})]


@given("a pipeline that has a component with only default inputs", target_fixture="pipeline_data")
def pipeline_that_has_a_component_with_only_default_inputs(pipeline_class):
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

    pipe = pipeline_class(max_runs_per_component=1)

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
                                query="What is the capital of France?",
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
                expected_component_calls={
                    ("answer_builder", 1): {
                        "documents": [
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
                        "meta": None,
                        "pattern": None,
                        "query": "What is the capital of France?",
                        "reference_pattern": None,
                        "replies": ["Paris"],
                    },
                    ("generator", 1): {
                        "prompt": "Given the following information, answer the "
                        "question.\n"
                        "Context:\n"
                        "    Paris is the capital of France\n"
                        "    Rome is the capital of Italy\n"
                        "Question: What is the capital of France?"
                    },
                    ("prompt_builder", 1): {
                        "documents": [
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
                        "query": "What is the capital of France?",
                        "template": None,
                        "template_variables": None,
                    },
                    ("retriever", 1): {
                        "filters": None,
                        "query": "What is the capital of France?",
                        "scale_score": None,
                        "top_k": None,
                    },
                },
            )
        ],
    )


@given(
    "a pipeline that has a component with only default inputs as first to run and receives inputs from a loop",
    target_fixture="pipeline_data",
)
def pipeline_that_has_a_component_with_only_default_inputs_as_first_to_run_and_receives_inputs_from_a_loop(
    pipeline_class,
):
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

    pipe = pipeline_class(max_runs_per_component=1)

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
                inputs={"prompt_builder": {"query": "What is the capital of Italy?"}},
                expected_outputs={"router": {"correct_replies": ["Rome"]}},
                expected_component_calls={
                    ("generator", 1): {
                        "generation_kwargs": None,
                        "prompt": "Answer the following question.\n\nQuestion: What is the capital of Italy?",
                    },
                    ("generator", 2): {
                        "generation_kwargs": None,
                        "prompt": "Answer the following question.\n"
                        "\n"
                        "Previously you replied incorrectly this:\n"
                        "\n"
                        " - Paris\n"
                        "\n"
                        "\n"
                        "Question: What is the capital of Italy?",
                    },
                    ("prompt_builder", 1): {
                        "previous_replies": "",
                        "query": "What is the capital of Italy?",
                        "template": None,
                        "template_variables": None,
                    },
                    ("prompt_builder", 2): {
                        "previous_replies": ["Paris"],
                        "query": "What is the capital of Italy?",
                        "template": None,
                        "template_variables": None,
                    },
                    ("router", 1): {"replies": ["Paris"]},
                    ("router", 2): {"replies": ["Rome"]},
                },
            )
        ],
    )


@given(
    "a pipeline that has multiple branches that merge into a component with a single variadic input",
    target_fixture="pipeline_data",
)
def pipeline_that_has_multiple_branches_that_merge_into_a_component_with_a_single_variadic_input(pipeline_class):
    pipeline = pipeline_class(max_runs_per_component=1)
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
                expected_component_calls={
                    ("add_one", 1): {"add": None, "value": 1},
                    ("add_ten", 1): {"add": None, "value": 2},
                    ("parity", 1): {"value": 2},
                    ("sum", 1): {"values": [2, 12]},
                },
            ),
            PipelineRunData(
                inputs={"add_one": {"value": 2}},
                expected_outputs={"sum": {"total": 17}},
                expected_component_calls={
                    ("add_four", 1): {"add": None, "value": 3},
                    ("add_one", 1): {"add": None, "value": 2},
                    ("add_one_again", 1): {"add": None, "value": 7},
                    ("double", 1): {"value": 3},
                    ("parity", 1): {"value": 3},
                    ("sum", 1): {"values": [3, 6, 8]},
                },
            ),
        ],
    )


@given(
    "a pipeline that has multiple branches of different lengths that merge into a component with a single variadic input",
    target_fixture="pipeline_data",
)
def pipeline_that_has_multiple_branches_of_different_lengths_that_merge_into_a_component_with_a_single_variadic_input(
    pipeline_class,
):
    pipeline = pipeline_class(max_runs_per_component=1)
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
                expected_component_calls={
                    ("first_addition", 1): {"add": None, "value": 1},
                    ("fourth_addition", 1): {"add": None, "value": 11},
                    ("second_addition", 1): {"add": None, "value": 3},
                    ("sum", 1): {"values": [3, 3, 5]},
                    ("third_addition", 1): {"add": None, "value": 1},
                },
            )
        ],
    )


@given("a pipeline that is linear and returns intermediate outputs", target_fixture="pipeline_data")
def pipeline_that_is_linear_and_returns_intermediate_outputs(pipeline_class):
    pipeline = pipeline_class(max_runs_per_component=1)
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
                expected_component_calls={
                    ("double", 1): {"value": 3},
                    ("first_addition", 1): {"add": None, "value": 1},
                    ("second_addition", 1): {"add": None, "value": 6},
                },
            ),
            PipelineRunData(
                inputs={"first_addition": {"value": 1}},
                include_outputs_from={"double"},
                expected_outputs={"double": {"value": 6}, "second_addition": {"result": 7}},
                expected_component_calls={
                    ("double", 1): {"value": 3},
                    ("first_addition", 1): {"add": None, "value": 1},
                    ("second_addition", 1): {"add": None, "value": 6},
                },
            ),
        ],
    )


@given("a pipeline that has a loop and returns intermediate outputs from it", target_fixture="pipeline_data")
def pipeline_that_has_a_loop_and_returns_intermediate_outputs_from_it(pipeline_class):
    pipeline = pipeline_class(max_runs_per_component=10)
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
                expected_component_calls={
                    ("accumulator", 1): {"value": 4},
                    ("accumulator", 2): {"value": 4},
                    ("add_one", 1): {"add": None, "value": 3},
                    ("add_three", 1): {"add": None, "value": 8},
                    ("add_two", 1): {"add": None, "value": 11},
                    ("below_10", 1): {"threshold": None, "value": 4},
                    ("below_10", 2): {"threshold": None, "value": 4},
                    ("below_10", 3): {"threshold": None, "value": 11},
                    ("below_5", 1): {"threshold": None, "value": 4},
                    ("below_5", 2): {"threshold": None, "value": 8},
                    ("branch_joiner", 1): {"value": [4]},
                    ("branch_joiner", 2): {"value": [4]},
                    ("branch_joiner", 3): {"value": [11]},
                },
            )
        ],
    )


@given(
    "a pipeline that is linear and returns intermediate outputs from multiple sockets", target_fixture="pipeline_data"
)
def pipeline_that_is_linear_and_returns_intermediate_outputs_from_multiple_sockets(pipeline_class):
    @component
    class DoubleWithOriginal:
        """
        Doubles the input value and returns the original value as well.
        """

        @component.output_types(value=int, original=int)
        def run(self, value: int):
            return {"value": value * 2, "original": value}

    pipeline = pipeline_class(max_runs_per_component=1)
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
                expected_component_calls={
                    ("double", 1): {"value": 3},
                    ("first_addition", 1): {"add": None, "value": 1},
                    ("second_addition", 1): {"add": None, "value": 6},
                },
            ),
            PipelineRunData(
                inputs={"first_addition": {"value": 1}},
                include_outputs_from={"double"},
                expected_outputs={"double": {"original": 3, "value": 6}, "second_addition": {"result": 7}},
                expected_component_calls={
                    ("double", 1): {"value": 3},
                    ("first_addition", 1): {"add": None, "value": 1},
                    ("second_addition", 1): {"add": None, "value": 6},
                },
            ),
        ],
    )


@given(
    "a pipeline that has a component with default inputs that doesn't receive anything from its sender",
    target_fixture="pipeline_data",
)
def pipeline_that_has_a_component_with_default_inputs_that_doesnt_receive_anything_from_its_sender(pipeline_class):
    routes = [
        {"condition": "{{'reisen' in sentence}}", "output": "German", "output_name": "language_1", "output_type": str},
        {"condition": "{{'viajar' in sentence}}", "output": "Spanish", "output_name": "language_2", "output_type": str},
    ]
    router = ConditionalRouter(routes)

    pipeline = pipeline_class(max_runs_per_component=1)
    pipeline.add_component("router", router)
    pipeline.add_component("pb", PromptBuilder(template="Ok, I know, that's {{language}}"))
    pipeline.connect("router.language_2", "pb.language")

    return (
        pipeline,
        [
            PipelineRunData(
                inputs={"router": {"sentence": "Wir mussen reisen"}},
                expected_outputs={"router": {"language_1": "German"}},
                expected_component_calls={("router", 1): {"sentence": "Wir mussen reisen"}},
            ),
            PipelineRunData(
                inputs={"router": {"sentence": "Yo tengo que viajar"}},
                expected_outputs={"pb": {"prompt": "Ok, I know, that's Spanish"}},
                expected_component_calls={
                    ("pb", 1): {"language": "Spanish", "template": None, "template_variables": None},
                    ("router", 1): {"sentence": "Yo tengo que viajar"},
                },
            ),
        ],
    )


@given(
    "a pipeline that has a component with default inputs that doesn't receive anything from its sender but receives input from user",
    target_fixture="pipeline_data",
)
def pipeline_that_has_a_component_with_default_inputs_that_doesnt_receive_anything_from_its_sender_but_receives_input_from_user(
    pipeline_class,
):
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

    pipeline = pipeline_class(max_runs_per_component=1)
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
                expected_component_calls={
                    ("fallback_llm", 1): {
                        "prompt": "User entered a query that cannot be answered "
                        "with the given table.\n"
                        "                    The query was: This is a "
                        "question with no_answer and the table had "
                        "columns: .\n"
                        "                    Let the user know why "
                        "the question cannot be answered"
                    },
                    ("fallback_prompt", 1): {
                        "columns": "",
                        "question": "This is a question with no_answer",
                        "template": None,
                        "template_variables": None,
                    },
                    ("llm", 1): {
                        "prompt": "Please generate an SQL query. The query should answer "
                        "the following Question: This is a question with "
                        "no_answer;\n"
                        "            If the question cannot be answered given "
                        "the provided table and columns, return 'no_answer'\n"
                        "            The query is to be answered for the table "
                        "is called 'absenteeism' with the following\n"
                        "            Columns: Age, Absenteeism_time_in_hours, "
                        "Days, Disciplinary_failure;\n"
                        "            Answer:"
                    },
                    ("prompt", 1): {
                        "columns": "Age, Absenteeism_time_in_hours, Days, Disciplinary_failure",
                        "question": "This is a question with no_answer",
                        "template": None,
                        "template_variables": None,
                    },
                    ("router", 1): {
                        "question": "This is a question with no_answer",
                        "replies": ["There's simply no_answer to this question"],
                    },
                },
            )
        ],
        [
            PipelineRunData(
                inputs={
                    "prompt": {"question": "This is a question that has an answer", "columns": columns},
                    "router": {"question": "This is a question that has an answer"},
                },
                expected_outputs={"sql_querier": {"results": "This is the query result", "query": "Some SQL query"}},
                expected_component_calls={
                    ("llm", 1): {
                        "prompt": "\n"
                        "    You are an experienced and accurate Turkish CX "
                        "speacialist that classifies customer comments into "
                        "pre-defined categories below:\n"
                        "\n"
                        "    Negative experience labels:\n"
                        "    - Late delivery\n"
                        "    - Rotten/spoilt item\n"
                        "    - Bad Courier behavior\n"
                        "\n"
                        "    Positive experience labels:\n"
                        "    - Good courier behavior\n"
                        "    - Thanks & appreciation\n"
                        "    - Love message to courier\n"
                        "    - Fast delivery\n"
                        "    - Quality of products\n"
                        "\n"
                        "    Create a JSON object as a response. The fields "
                        "are: 'positive_experience', 'negative_experience'.\n"
                        "    Assign at least one of the pre-defined labels to "
                        "the given customer comment under positive and "
                        "negative experience fields.\n"
                        "    If the comment has a positive experience, list "
                        "the label under 'positive_experience' field.\n"
                        "    If the comments has a negative_experience, list "
                        "it under the 'negative_experience' field.\n"
                        "    Here is the comment:\n"
                        "I loved the quality of the meal but the courier was "
                        "rude\n"
                        ". Just return the category names in the list. If "
                        "there aren't any, return an empty list.\n"
                        "\n"
                        "    \n"
                        "    "
                    },
                    ("llm", 2): {
                        "prompt": "\n"
                        "    You are an experienced and accurate Turkish CX "
                        "speacialist that classifies customer comments into "
                        "pre-defined categories below:\n"
                        "\n"
                        "    Negative experience labels:\n"
                        "    - Late delivery\n"
                        "    - Rotten/spoilt item\n"
                        "    - Bad Courier behavior\n"
                        "\n"
                        "    Positive experience labels:\n"
                        "    - Good courier behavior\n"
                        "    - Thanks & appreciation\n"
                        "    - Love message to courier\n"
                        "    - Fast delivery\n"
                        "    - Quality of products\n"
                        "\n"
                        "    Create a JSON object as a response. The fields "
                        "are: 'positive_experience', 'negative_experience'.\n"
                        "    Assign at least one of the pre-defined labels to "
                        "the given customer comment under positive and "
                        "negative experience fields.\n"
                        "    If the comment has a positive experience, list "
                        "the label under 'positive_experience' field.\n"
                        "    If the comments has a negative_experience, list "
                        "it under the 'negative_experience' field.\n"
                        "    Here is the comment:\n"
                        "I loved the quality of the meal but the courier was "
                        "rude\n"
                        ". Just return the category names in the list. If "
                        "there aren't any, return an empty list.\n"
                        "\n"
                        "    \n"
                        "    You already created the following output in a "
                        "previous attempt: ['This is an invalid reply']\n"
                        "    However, this doesn't comply with the format "
                        "requirements from above and triggered this Python "
                        "exception: this is an error message\n"
                        "    Correct the output and try again. Just return the "
                        "corrected output without any extra explanations.\n"
                        "    \n"
                        "    "
                    },
                    ("output_validator", 1): {"replies": ["This is a valid reply"]},
                    ("output_validator", 2): {"replies": ["This is a valid reply"]},
                    ("prompt_builder", 1): {
                        "comment": "",
                        "error_message": "",
                        "invalid_replies": "",
                        "template": None,
                        "template_variables": {"comment": "I loved the quality of the meal but the courier was rude"},
                    },
                    ("prompt_builder", 2): {
                        "comment": "",
                        "error_message": "this is an error message",
                        "invalid_replies": ["This is an invalid reply"],
                        "template": None,
                        "template_variables": {"comment": "I loved the quality of the meal but the courier was rude"},
                    },
                },
            )
        ],
    )


@given(
    "a pipeline that has a loop and a component with default inputs that doesn't receive anything from its sender but receives input from user",
    target_fixture="pipeline_data",
)
def pipeline_that_has_a_loop_and_a_component_with_default_inputs_that_doesnt_receive_anything_from_its_sender_but_receives_input_from_user(
    pipeline_class,
):
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

    pipeline = pipeline_class(max_runs_per_component=1)
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
                expected_component_calls={
                    ("llm", 1): {
                        "prompt": "\n"
                        "    You are an experienced and accurate Turkish CX "
                        "speacialist that classifies customer comments into "
                        "pre-defined categories below:\n"
                        "\n"
                        "    Negative experience labels:\n"
                        "    - Late delivery\n"
                        "    - Rotten/spoilt item\n"
                        "    - Bad Courier behavior\n"
                        "\n"
                        "    Positive experience labels:\n"
                        "    - Good courier behavior\n"
                        "    - Thanks & appreciation\n"
                        "    - Love message to courier\n"
                        "    - Fast delivery\n"
                        "    - Quality of products\n"
                        "\n"
                        "    Create a JSON object as a response. The fields "
                        "are: 'positive_experience', 'negative_experience'.\n"
                        "    Assign at least one of the pre-defined labels to "
                        "the given customer comment under positive and "
                        "negative experience fields.\n"
                        "    If the comment has a positive experience, list "
                        "the label under 'positive_experience' field.\n"
                        "    If the comments has a negative_experience, list "
                        "it under the 'negative_experience' field.\n"
                        "    Here is the comment:\n"
                        "I loved the quality of the meal but the courier was "
                        "rude\n"
                        ". Just return the category names in the list. If "
                        "there aren't any, return an empty list.\n"
                        "\n"
                        "    \n"
                        "    "
                    },
                    ("llm", 2): {
                        "prompt": "\n"
                        "    You are an experienced and accurate Turkish CX "
                        "speacialist that classifies customer comments into "
                        "pre-defined categories below:\n"
                        "\n"
                        "    Negative experience labels:\n"
                        "    - Late delivery\n"
                        "    - Rotten/spoilt item\n"
                        "    - Bad Courier behavior\n"
                        "\n"
                        "    Positive experience labels:\n"
                        "    - Good courier behavior\n"
                        "    - Thanks & appreciation\n"
                        "    - Love message to courier\n"
                        "    - Fast delivery\n"
                        "    - Quality of products\n"
                        "\n"
                        "    Create a JSON object as a response. The fields "
                        "are: 'positive_experience', 'negative_experience'.\n"
                        "    Assign at least one of the pre-defined labels to "
                        "the given customer comment under positive and "
                        "negative experience fields.\n"
                        "    If the comment has a positive experience, list "
                        "the label under 'positive_experience' field.\n"
                        "    If the comments has a negative_experience, list "
                        "it under the 'negative_experience' field.\n"
                        "    Here is the comment:\n"
                        "I loved the quality of the meal but the courier was "
                        "rude\n"
                        ". Just return the category names in the list. If "
                        "there aren't any, return an empty list.\n"
                        "\n"
                        "    \n"
                        "    You already created the following output in a "
                        "previous attempt: ['This is an invalid reply']\n"
                        "    However, this doesn't comply with the format "
                        "requirements from above and triggered this Python "
                        "exception: this is an error message\n"
                        "    Correct the output and try again. Just return the "
                        "corrected output without any extra explanations.\n"
                        "    \n"
                        "    "
                    },
                    ("output_validator", 1): {"replies": ["This is a valid reply"]},
                    ("output_validator", 2): {"replies": ["This is a valid reply"]},
                    ("prompt_builder", 1): {
                        "comment": "",
                        "error_message": "",
                        "invalid_replies": "",
                        "template": None,
                        "template_variables": {"comment": "I loved the quality of the meal but the courier was rude"},
                    },
                    ("prompt_builder", 2): {
                        "comment": "",
                        "error_message": "this is an error message",
                        "invalid_replies": ["This is an invalid reply"],
                        "template": None,
                        "template_variables": {"comment": "I loved the quality of the meal but the courier was rude"},
                    },
                },
            )
        ],
    )


@given(
    "a pipeline that has multiple components with only default inputs and are added in a different order from the order of execution",
    target_fixture="pipeline_data",
)
def pipeline_that_has_multiple_components_with_only_default_inputs_and_are_added_in_a_different_order_from_the_order_of_execution(
    pipeline_class,
):
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

    pipeline = pipeline_class(max_runs_per_component=1)
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
                expected_component_calls={
                    ("llm", 1): {
                        "generation_kwargs": None,
                        "prompt": "\n"
                        "    According to these documents:\n"
                        "\n"
                        "    \n"
                        "    This is a document\n"
                        "    \n"
                        "\n"
                        "    Answer the given question: \n"
                        "    \n"
                        "    This is a reply\n"
                        "    \n"
                        "    \n"
                        "    Answer:\n"
                        "    ",
                    },
                    ("prompt_builder1", 1): {
                        "question": "Wha i Acromegaly?",
                        "template": None,
                        "template_variables": None,
                    },
                    ("prompt_builder2", 1): {
                        "documents": [
                            Document(
                                id="9d51914541072d3d822910785727db8a3838dba5ca6ebb0a543969260ecdeda6",
                                content="This is a document",
                            )
                        ],
                        "question": "\n    \n    This is a reply\n    \n    ",
                        "template": None,
                        "template_variables": None,
                    },
                    ("prompt_builder3", 1): {
                        "replies": ["This is a reply"],
                        "template": None,
                        "template_variables": None,
                    },
                    ("ranker", 1): {
                        "calibration_factor": None,
                        "documents": [
                            Document(
                                id="9d51914541072d3d822910785727db8a3838dba5ca6ebb0a543969260ecdeda6",
                                content="This is a document",
                            )
                        ],
                        "query": "\n    \n    This is a reply\n    \n    ",
                        "scale_score": None,
                        "score_threshold": None,
                        "top_k": None,
                    },
                    ("retriever", 1): {
                        "filters": None,
                        "query": "\n    \n    This is a reply\n    \n    ",
                        "scale_score": None,
                        "top_k": None,
                    },
                    ("spellchecker", 1): {
                        "generation_kwargs": None,
                        "prompt": "\n"
                        "    You are a spellchecking system. Check "
                        "the given query and fill in the corrected "
                        "query.\n"
                        "\n"
                        "    Question: Wha i Acromegaly?\n"
                        "    Corrected question:\n"
                        "    ",
                    },
                },
            )
        ],
    )


@given("a pipeline that is linear with conditional branching and multiple joins", target_fixture="pipeline_data")
def that_is_linear_with_conditional_branching_and_multiple_joins(pipeline_class):
    pipeline = pipeline_class()

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
                expected_component_calls={
                    ("router", 1): {"query": "I'm a legit question"},
                    ("text_embedder", 1): {"text": "I'm a legit question"},
                    ("bm25retriever", 1): {"query": "I'm a legit question"},
                    ("retriever", 1): {"query_embedding": [1.0, 2.0, 3.0]},
                    ("joinerhybrid", 1): {
                        "documents": [
                            [Document(content="This is a document")],
                            [Document(content="This is another document")],
                        ],
                        "top_k": None,
                    },
                    ("ranker", 1): {
                        "query": "I'm a legit question",
                        "documents": [
                            Document(content="This is a document"),
                            Document(content="This is another document"),
                        ],
                    },
                    ("joinerfinal", 1): {
                        "documents": [
                            [Document(content="This is a document"), Document(content="This is another document")]
                        ],
                        "top_k": None,
                    },
                },
            ),
            PipelineRunData(
                inputs={"router": {"query": "I'm a nasty prompt injection"}},
                expected_outputs={"joinerfinal": {"documents": []}},
                expected_component_calls={
                    ("router", 1): {"query": "I'm a nasty prompt injection"},
                    ("emptyretriever", 1): {"query": "I'm a nasty prompt injection"},
                    ("joinerfinal", 1): {"documents": [[]], "top_k": None},
                },
            ),
        ],
    )


@given("a pipeline that is a simple agent", target_fixture="pipeline_data")
def that_is_a_simple_agent(pipeline_class):
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
    pipeline = pipeline_class()
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
            expected_component_calls={
                ("llm", 1): {
                    "generation_kwargs": None,
                    "messages": [
                        ChatMessage(
                            _role="user",
                            _content=[
                                TextContent(
                                    text="\n    Solve a question answering task with interleaving Thought, Action, Observation steps.\n\n    Thought reasons about the current situation\n\n    Action can be:\n    google_search - Searches Google for the exact concept/entity (given in square brackets) and returns the results for you to use\n    finish - Returns the final answer (given in square brackets) and finishes the task\n\n    Observation summarizes the Action outcome and helps in formulating the next\n    Thought in Thought, Action, Observation interleaving triplet of steps.\n\n    After each Observation, provide the next Thought and next Action.\n    Don't execute multiple steps even though you know the answer.\n    Only generate Thought and Action, never Observation, you'll get Observation from Action.\n    Follow the pattern in the example below.\n\n    Example:\n    ###########################\n    Question: Which magazine was started first Arthur’s Magazine or First for Women?\n    Thought: I need to search Arthur’s Magazine and First for Women, and find which was started\n    first.\n    Action: google_search[When was 'Arthur’s Magazine' started?]\n    Observation: Arthur’s Magazine was an American literary periodical ˘\n    published in Philadelphia and founded in 1844. Edited by Timothy Shay Arthur, it featured work by\n    Edgar A. Poe, J.H. Ingraham, Sarah Josepha Hale, Thomas G. Spear, and others. In May 1846\n    it was merged into Godey’s Lady’s Book.\n    Thought: Arthur’s Magazine was started in 1844. I need to search First for Women founding date next\n    Action: google_search[When was 'First for Women' magazine started?]\n    Observation: First for Women is a woman’s magazine published by Bauer Media Group in the\n    USA. The magazine was started in 1989. It is based in Englewood Cliffs, New Jersey. In 2011\n    the circulation of the magazine was 1,310,696 copies.\n    Thought: First for Women was started in 1989. 1844 (Arthur’s Magazine) ¡ 1989 (First for\n    Women), so Arthur’s Magazine was started first.\n    Action: finish[Arthur’s Magazine]\n    ############################\n\n    Let's start, the question is: which tower is taller: eiffel tower or tower of pisa?\n\n    Thought:\n    "
                                )
                            ],
                            _name=None,
                            _meta={},
                        )
                    ],
                },
                ("llm", 2): {
                    "generation_kwargs": None,
                    "messages": [
                        ChatMessage(
                            _role="user",
                            _content=[
                                TextContent(
                                    text="\n    Solve a question answering task with interleaving Thought, Action, Observation steps.\n\n    Thought reasons about the current situation\n\n    Action can be:\n    google_search - Searches Google for the exact concept/entity (given in square brackets) and returns the results for you to use\n    finish - Returns the final answer (given in square brackets) and finishes the task\n\n    Observation summarizes the Action outcome and helps in formulating the next\n    Thought in Thought, Action, Observation interleaving triplet of steps.\n\n    After each Observation, provide the next Thought and next Action.\n    Don't execute multiple steps even though you know the answer.\n    Only generate Thought and Action, never Observation, you'll get Observation from Action.\n    Follow the pattern in the example below.\n\n    Example:\n    ###########################\n    Question: Which magazine was started first Arthur’s Magazine or First for Women?\n    Thought: I need to search Arthur’s Magazine and First for Women, and find which was started\n    first.\n    Action: google_search[When was 'Arthur’s Magazine' started?]\n    Observation: Arthur’s Magazine was an American literary periodical ˘\n    published in Philadelphia and founded in 1844. Edited by Timothy Shay Arthur, it featured work by\n    Edgar A. Poe, J.H. Ingraham, Sarah Josepha Hale, Thomas G. Spear, and others. In May 1846\n    it was merged into Godey’s Lady’s Book.\n    Thought: Arthur’s Magazine was started in 1844. I need to search First for Women founding date next\n    Action: google_search[When was 'First for Women' magazine started?]\n    Observation: First for Women is a woman’s magazine published by Bauer Media Group in the\n    USA. The magazine was started in 1989. It is based in Englewood Cliffs, New Jersey. In 2011\n    the circulation of the magazine was 1,310,696 copies.\n    Thought: First for Women was started in 1989. 1844 (Arthur’s Magazine) ¡ 1989 (First for\n    Women), so Arthur’s Magazine was started first.\n    Action: finish[Arthur’s Magazine]\n    ############################\n\n    Let's start, the question is: which tower is taller: eiffel tower or tower of pisa?\n\n    Thought:\n    thinking\n Action: google_search[What is taller, Eiffel Tower or Leaning Tower of Pisa]\nObservation: Tower of Pisa is 55 meters tall\n\n\nThought: "
                                )
                            ],
                            _name=None,
                            _meta={},
                        )
                    ],
                },
                ("main_input", 1): {
                    "value": [
                        [
                            ChatMessage(
                                _role="user",
                                _content=[
                                    TextContent(
                                        text="\n    Solve a question answering task with interleaving Thought, Action, Observation steps.\n\n    Thought reasons about the current situation\n\n    Action can be:\n    google_search - Searches Google for the exact concept/entity (given in square brackets) and returns the results for you to use\n    finish - Returns the final answer (given in square brackets) and finishes the task\n\n    Observation summarizes the Action outcome and helps in formulating the next\n    Thought in Thought, Action, Observation interleaving triplet of steps.\n\n    After each Observation, provide the next Thought and next Action.\n    Don't execute multiple steps even though you know the answer.\n    Only generate Thought and Action, never Observation, you'll get Observation from Action.\n    Follow the pattern in the example below.\n\n    Example:\n    ###########################\n    Question: Which magazine was started first Arthur’s Magazine or First for Women?\n    Thought: I need to search Arthur’s Magazine and First for Women, and find which was started\n    first.\n    Action: google_search[When was 'Arthur’s Magazine' started?]\n    Observation: Arthur’s Magazine was an American literary periodical ˘\n    published in Philadelphia and founded in 1844. Edited by Timothy Shay Arthur, it featured work by\n    Edgar A. Poe, J.H. Ingraham, Sarah Josepha Hale, Thomas G. Spear, and others. In May 1846\n    it was merged into Godey’s Lady’s Book.\n    Thought: Arthur’s Magazine was started in 1844. I need to search First for Women founding date next\n    Action: google_search[When was 'First for Women' magazine started?]\n    Observation: First for Women is a woman’s magazine published by Bauer Media Group in the\n    USA. The magazine was started in 1989. It is based in Englewood Cliffs, New Jersey. In 2011\n    the circulation of the magazine was 1,310,696 copies.\n    Thought: First for Women was started in 1989. 1844 (Arthur’s Magazine) ¡ 1989 (First for\n    Women), so Arthur’s Magazine was started first.\n    Action: finish[Arthur’s Magazine]\n    ############################\n\n    Let's start, the question is: {{query}}\n\n    Thought:\n    "
                                    )
                                ],
                                _name=None,
                                _meta={},
                            )
                        ]
                    ]
                },
                ("main_input", 2): {
                    "value": [
                        [
                            ChatMessage(
                                _role="user",
                                _content=[
                                    TextContent(
                                        text="\n    Solve a question answering task with interleaving Thought, Action, Observation steps.\n\n    Thought reasons about the current situation\n\n    Action can be:\n    google_search - Searches Google for the exact concept/entity (given in square brackets) and returns the results for you to use\n    finish - Returns the final answer (given in square brackets) and finishes the task\n\n    Observation summarizes the Action outcome and helps in formulating the next\n    Thought in Thought, Action, Observation interleaving triplet of steps.\n\n    After each Observation, provide the next Thought and next Action.\n    Don't execute multiple steps even though you know the answer.\n    Only generate Thought and Action, never Observation, you'll get Observation from Action.\n    Follow the pattern in the example below.\n\n    Example:\n    ###########################\n    Question: Which magazine was started first Arthur’s Magazine or First for Women?\n    Thought: I need to search Arthur’s Magazine and First for Women, and find which was started\n    first.\n    Action: google_search[When was 'Arthur’s Magazine' started?]\n    Observation: Arthur’s Magazine was an American literary periodical ˘\n    published in Philadelphia and founded in 1844. Edited by Timothy Shay Arthur, it featured work by\n    Edgar A. Poe, J.H. Ingraham, Sarah Josepha Hale, Thomas G. Spear, and others. In May 1846\n    it was merged into Godey’s Lady’s Book.\n    Thought: Arthur’s Magazine was started in 1844. I need to search First for Women founding date next\n    Action: google_search[When was 'First for Women' magazine started?]\n    Observation: First for Women is a woman’s magazine published by Bauer Media Group in the\n    USA. The magazine was started in 1989. It is based in Englewood Cliffs, New Jersey. In 2011\n    the circulation of the magazine was 1,310,696 copies.\n    Thought: First for Women was started in 1989. 1844 (Arthur’s Magazine) ¡ 1989 (First for\n    Women), so Arthur’s Magazine was started first.\n    Action: finish[Arthur’s Magazine]\n    ############################\n\n    Let's start, the question is: which tower is taller: eiffel tower or tower of pisa?\n\n    Thought:\n    thinking\n Action: google_search[What is taller, Eiffel Tower or Leaning Tower of Pisa]\nObservation: Tower of Pisa is 55 meters tall\n\n\nThought: "
                                    )
                                ],
                                _name=None,
                                _meta={},
                            )
                        ]
                    ]
                },
                ("prompt_builder", 1): {
                    "query": "which tower is taller: eiffel tower or tower of pisa?",
                    "template": [
                        ChatMessage(
                            _role="user",
                            _content=[
                                TextContent(
                                    text="\n    Solve a question answering task with interleaving Thought, Action, Observation steps.\n\n    Thought reasons about the current situation\n\n    Action can be:\n    google_search - Searches Google for the exact concept/entity (given in square brackets) and returns the results for you to use\n    finish - Returns the final answer (given in square brackets) and finishes the task\n\n    Observation summarizes the Action outcome and helps in formulating the next\n    Thought in Thought, Action, Observation interleaving triplet of steps.\n\n    After each Observation, provide the next Thought and next Action.\n    Don't execute multiple steps even though you know the answer.\n    Only generate Thought and Action, never Observation, you'll get Observation from Action.\n    Follow the pattern in the example below.\n\n    Example:\n    ###########################\n    Question: Which magazine was started first Arthur’s Magazine or First for Women?\n    Thought: I need to search Arthur’s Magazine and First for Women, and find which was started\n    first.\n    Action: google_search[When was 'Arthur’s Magazine' started?]\n    Observation: Arthur’s Magazine was an American literary periodical ˘\n    published in Philadelphia and founded in 1844. Edited by Timothy Shay Arthur, it featured work by\n    Edgar A. Poe, J.H. Ingraham, Sarah Josepha Hale, Thomas G. Spear, and others. In May 1846\n    it was merged into Godey’s Lady’s Book.\n    Thought: Arthur’s Magazine was started in 1844. I need to search First for Women founding date next\n    Action: google_search[When was 'First for Women' magazine started?]\n    Observation: First for Women is a woman’s magazine published by Bauer Media Group in the\n    USA. The magazine was started in 1989. It is based in Englewood Cliffs, New Jersey. In 2011\n    the circulation of the magazine was 1,310,696 copies.\n    Thought: First for Women was started in 1989. 1844 (Arthur’s Magazine) ¡ 1989 (First for\n    Women), so Arthur’s Magazine was started first.\n    Action: finish[Arthur’s Magazine]\n    ############################\n\n    Let's start, the question is: {{query}}\n\n    Thought:\n    "
                                )
                            ],
                            _name=None,
                            _meta={},
                        )
                    ],
                    "template_variables": None,
                },
                ("prompt_builder", 2): {
                    "query": "which tower is taller: eiffel tower or tower of pisa?",
                    "template": [
                        ChatMessage(
                            _role="user",
                            _content=[
                                TextContent(
                                    text="\n    Solve a question answering task with interleaving Thought, Action, Observation steps.\n\n    Thought reasons about the current situation\n\n    Action can be:\n    google_search - Searches Google for the exact concept/entity (given in square brackets) and returns the results for you to use\n    finish - Returns the final answer (given in square brackets) and finishes the task\n\n    Observation summarizes the Action outcome and helps in formulating the next\n    Thought in Thought, Action, Observation interleaving triplet of steps.\n\n    After each Observation, provide the next Thought and next Action.\n    Don't execute multiple steps even though you know the answer.\n    Only generate Thought and Action, never Observation, you'll get Observation from Action.\n    Follow the pattern in the example below.\n\n    Example:\n    ###########################\n    Question: Which magazine was started first Arthur’s Magazine or First for Women?\n    Thought: I need to search Arthur’s Magazine and First for Women, and find which was started\n    first.\n    Action: google_search[When was 'Arthur’s Magazine' started?]\n    Observation: Arthur’s Magazine was an American literary periodical ˘\n    published in Philadelphia and founded in 1844. Edited by Timothy Shay Arthur, it featured work by\n    Edgar A. Poe, J.H. Ingraham, Sarah Josepha Hale, Thomas G. Spear, and others. In May 1846\n    it was merged into Godey’s Lady’s Book.\n    Thought: Arthur’s Magazine was started in 1844. I need to search First for Women founding date next\n    Action: google_search[When was 'First for Women' magazine started?]\n    Observation: First for Women is a woman’s magazine published by Bauer Media Group in the\n    USA. The magazine was started in 1989. It is based in Englewood Cliffs, New Jersey. In 2011\n    the circulation of the magazine was 1,310,696 copies.\n    Thought: First for Women was started in 1989. 1844 (Arthur’s Magazine) ¡ 1989 (First for\n    Women), so Arthur’s Magazine was started first.\n    Action: finish[Arthur’s Magazine]\n    ############################\n\n    Let's start, the question is: which tower is taller: eiffel tower or tower of pisa?\n\n    Thought:\n    thinking\n Action: google_search[What is taller, Eiffel Tower or Leaning Tower of Pisa]\nObservation: Tower of Pisa is 55 meters tall\n\n\nThought: "
                                )
                            ],
                            _name=None,
                            _meta={},
                        )
                    ],
                    "template_variables": None,
                },
                ("prompt_concatenator_after_action", 1): {
                    "current_prompt": [
                        ChatMessage(
                            _role="user",
                            _content=[
                                TextContent(
                                    text="\n    Solve a question answering task with interleaving Thought, Action, Observation steps.\n\n    Thought reasons about the current situation\n\n    Action can be:\n    google_search - Searches Google for the exact concept/entity (given in square brackets) and returns the results for you to use\n    finish - Returns the final answer (given in square brackets) and finishes the task\n\n    Observation summarizes the Action outcome and helps in formulating the next\n    Thought in Thought, Action, Observation interleaving triplet of steps.\n\n    After each Observation, provide the next Thought and next Action.\n    Don't execute multiple steps even though you know the answer.\n    Only generate Thought and Action, never Observation, you'll get Observation from Action.\n    Follow the pattern in the example below.\n\n    Example:\n    ###########################\n    Question: Which magazine was started first Arthur’s Magazine or First for Women?\n    Thought: I need to search Arthur’s Magazine and First for Women, and find which was started\n    first.\n    Action: google_search[When was 'Arthur’s Magazine' started?]\n    Observation: Arthur’s Magazine was an American literary periodical ˘\n    published in Philadelphia and founded in 1844. Edited by Timothy Shay Arthur, it featured work by\n    Edgar A. Poe, J.H. Ingraham, Sarah Josepha Hale, Thomas G. Spear, and others. In May 1846\n    it was merged into Godey’s Lady’s Book.\n    Thought: Arthur’s Magazine was started in 1844. I need to search First for Women founding date next\n    Action: google_search[When was 'First for Women' magazine started?]\n    Observation: First for Women is a woman’s magazine published by Bauer Media Group in the\n    USA. The magazine was started in 1989. It is based in Englewood Cliffs, New Jersey. In 2011\n    the circulation of the magazine was 1,310,696 copies.\n    Thought: First for Women was started in 1989. 1844 (Arthur’s Magazine) ¡ 1989 (First for\n    Women), so Arthur’s Magazine was started first.\n    Action: finish[Arthur’s Magazine]\n    ############################\n\n    Let's start, the question is: which tower is taller: eiffel tower or tower of pisa?\n\n    Thought:\n    "
                                )
                            ],
                            _name=None,
                            _meta={},
                        )
                    ],
                    "replies": [
                        ChatMessage(
                            _role="assistant",
                            _content=[
                                TextContent(
                                    text="thinking\n Action: google_search[What is taller, Eiffel Tower or Leaning Tower of Pisa]\n"
                                )
                            ],
                            _name=None,
                            _meta={},
                        )
                    ],
                },
                ("prompt_concatenator_after_action", 2): {
                    "current_prompt": [
                        ChatMessage(
                            _role="user",
                            _content=[
                                TextContent(
                                    text="\n    Solve a question answering task with interleaving Thought, Action, Observation steps.\n\n    Thought reasons about the current situation\n\n    Action can be:\n    google_search - Searches Google for the exact concept/entity (given in square brackets) and returns the results for you to use\n    finish - Returns the final answer (given in square brackets) and finishes the task\n\n    Observation summarizes the Action outcome and helps in formulating the next\n    Thought in Thought, Action, Observation interleaving triplet of steps.\n\n    After each Observation, provide the next Thought and next Action.\n    Don't execute multiple steps even though you know the answer.\n    Only generate Thought and Action, never Observation, you'll get Observation from Action.\n    Follow the pattern in the example below.\n\n    Example:\n    ###########################\n    Question: Which magazine was started first Arthur’s Magazine or First for Women?\n    Thought: I need to search Arthur’s Magazine and First for Women, and find which was started\n    first.\n    Action: google_search[When was 'Arthur’s Magazine' started?]\n    Observation: Arthur’s Magazine was an American literary periodical ˘\n    published in Philadelphia and founded in 1844. Edited by Timothy Shay Arthur, it featured work by\n    Edgar A. Poe, J.H. Ingraham, Sarah Josepha Hale, Thomas G. Spear, and others. In May 1846\n    it was merged into Godey’s Lady’s Book.\n    Thought: Arthur’s Magazine was started in 1844. I need to search First for Women founding date next\n    Action: google_search[When was 'First for Women' magazine started?]\n    Observation: First for Women is a woman’s magazine published by Bauer Media Group in the\n    USA. The magazine was started in 1989. It is based in Englewood Cliffs, New Jersey. In 2011\n    the circulation of the magazine was 1,310,696 copies.\n    Thought: First for Women was started in 1989. 1844 (Arthur’s Magazine) ¡ 1989 (First for\n    Women), so Arthur’s Magazine was started first.\n    Action: finish[Arthur’s Magazine]\n    ############################\n\n    Let's start, the question is: which tower is taller: eiffel tower or tower of pisa?\n\n    Thought:\n    thinking\n Action: google_search[What is taller, Eiffel Tower or Leaning Tower of Pisa]\nObservation: Tower of Pisa is 55 meters tall\n\n\nThought: "
                                )
                            ],
                            _name=None,
                            _meta={},
                        )
                    ],
                    "replies": [
                        ChatMessage(
                            _role="assistant",
                            _content=[TextContent(text="thinking\n Action: finish[Eiffel Tower]\n")],
                            _name=None,
                            _meta={},
                        )
                    ],
                },
                ("prompt_concatenator_after_observation", 1): {
                    "current_prompt": [
                        ChatMessage(
                            _role="user",
                            _content=[
                                TextContent(
                                    text="\n    Solve a question answering task with interleaving Thought, Action, Observation steps.\n\n    Thought reasons about the current situation\n\n    Action can be:\n    google_search - Searches Google for the exact concept/entity (given in square brackets) and returns the results for you to use\n    finish - Returns the final answer (given in square brackets) and finishes the task\n\n    Observation summarizes the Action outcome and helps in formulating the next\n    Thought in Thought, Action, Observation interleaving triplet of steps.\n\n    After each Observation, provide the next Thought and next Action.\n    Don't execute multiple steps even though you know the answer.\n    Only generate Thought and Action, never Observation, you'll get Observation from Action.\n    Follow the pattern in the example below.\n\n    Example:\n    ###########################\n    Question: Which magazine was started first Arthur’s Magazine or First for Women?\n    Thought: I need to search Arthur’s Magazine and First for Women, and find which was started\n    first.\n    Action: google_search[When was 'Arthur’s Magazine' started?]\n    Observation: Arthur’s Magazine was an American literary periodical ˘\n    published in Philadelphia and founded in 1844. Edited by Timothy Shay Arthur, it featured work by\n    Edgar A. Poe, J.H. Ingraham, Sarah Josepha Hale, Thomas G. Spear, and others. In May 1846\n    it was merged into Godey’s Lady’s Book.\n    Thought: Arthur’s Magazine was started in 1844. I need to search First for Women founding date next\n    Action: google_search[When was 'First for Women' magazine started?]\n    Observation: First for Women is a woman’s magazine published by Bauer Media Group in the\n    USA. The magazine was started in 1989. It is based in Englewood Cliffs, New Jersey. In 2011\n    the circulation of the magazine was 1,310,696 copies.\n    Thought: First for Women was started in 1989. 1844 (Arthur’s Magazine) ¡ 1989 (First for\n    Women), so Arthur’s Magazine was started first.\n    Action: finish[Arthur’s Magazine]\n    ############################\n\n    Let's start, the question is: which tower is taller: eiffel tower or tower of pisa?\n\n    Thought:\n    thinking\n Action: google_search[What is taller, Eiffel Tower or Leaning Tower of Pisa]\n"
                                )
                            ],
                            _name=None,
                            _meta={},
                        )
                    ],
                    "replies": [
                        ChatMessage(
                            _role="assistant",
                            _content=[TextContent(text="Observation: Tower of Pisa is 55 meters tall\n\n")],
                            _name=None,
                            _meta={},
                        )
                    ],
                },
                ("router", 1): {
                    "tool_id_and_param": ["google_search", "What is taller, Eiffel Tower or Leaning Tower of Pisa"]
                },
                ("router", 2): {"tool_id_and_param": ["finish", "Eiffel Tower"]},
                ("router_search", 1): {"query": "What is taller, Eiffel Tower or Leaning Tower of Pisa"},
                ("search_llm", 1): {
                    "generation_kwargs": None,
                    "messages": [
                        ChatMessage(
                            _role="user",
                            _content=[
                                TextContent(
                                    text="\n    Given these web search results:\n\n    \n        Eiffel Tower is 300 meters tall\n    \n        Tower of Pisa is 55 meters tall\n    \n\n    Be as brief as possible, max one sentence.\n    Answer the question: What is taller, Eiffel Tower or Leaning Tower of Pisa\n    "
                                )
                            ],
                            _name=None,
                            _meta={},
                        )
                    ],
                },
                ("search_output_adapter", 1): {
                    "replies": [
                        ChatMessage(
                            _role="assistant",
                            _content=[TextContent(text="Tower of Pisa is 55 meters tall\n")],
                            _name=None,
                            _meta={},
                        )
                    ]
                },
                ("search_prompt_builder", 1): {
                    "documents": [
                        Document(
                            id="c37eb19352b261b17314cac9e1539921b5996f88c99ad0b134f12effb38ed467",
                            content="Eiffel Tower is 300 meters tall",
                        ),
                        Document(
                            id="c5281056a220c32e6fa1c4ae7d3f263c0f25fd620592c5e45049a9dcb778f129",
                            content="Tower of Pisa is 55 meters tall",
                        ),
                    ],
                    "search_query": "What is taller, Eiffel Tower or Leaning Tower of Pisa",
                    "template": [
                        ChatMessage(
                            _role="user",
                            _content=[
                                TextContent(
                                    text="\n    Given these web search results:\n\n    {% for doc in documents %}\n        {{ doc.content }}\n    {% endfor %}\n\n    Be as brief as possible, max one sentence.\n    Answer the question: {{search_query}}\n    "
                                )
                            ],
                            _name=None,
                            _meta={},
                        )
                    ],
                    "template_variables": None,
                },
                ("tool_extractor", 1): {
                    "messages": [
                        ChatMessage(
                            _role="user",
                            _content=[
                                TextContent(
                                    text="\n    Solve a question answering task with interleaving Thought, Action, Observation steps.\n\n    Thought reasons about the current situation\n\n    Action can be:\n    google_search - Searches Google for the exact concept/entity (given in square brackets) and returns the results for you to use\n    finish - Returns the final answer (given in square brackets) and finishes the task\n\n    Observation summarizes the Action outcome and helps in formulating the next\n    Thought in Thought, Action, Observation interleaving triplet of steps.\n\n    After each Observation, provide the next Thought and next Action.\n    Don't execute multiple steps even though you know the answer.\n    Only generate Thought and Action, never Observation, you'll get Observation from Action.\n    Follow the pattern in the example below.\n\n    Example:\n    ###########################\n    Question: Which magazine was started first Arthur’s Magazine or First for Women?\n    Thought: I need to search Arthur’s Magazine and First for Women, and find which was started\n    first.\n    Action: google_search[When was 'Arthur’s Magazine' started?]\n    Observation: Arthur’s Magazine was an American literary periodical ˘\n    published in Philadelphia and founded in 1844. Edited by Timothy Shay Arthur, it featured work by\n    Edgar A. Poe, J.H. Ingraham, Sarah Josepha Hale, Thomas G. Spear, and others. In May 1846\n    it was merged into Godey’s Lady’s Book.\n    Thought: Arthur’s Magazine was started in 1844. I need to search First for Women founding date next\n    Action: google_search[When was 'First for Women' magazine started?]\n    Observation: First for Women is a woman’s magazine published by Bauer Media Group in the\n    USA. The magazine was started in 1989. It is based in Englewood Cliffs, New Jersey. In 2011\n    the circulation of the magazine was 1,310,696 copies.\n    Thought: First for Women was started in 1989. 1844 (Arthur’s Magazine) ¡ 1989 (First for\n    Women), so Arthur’s Magazine was started first.\n    Action: finish[Arthur’s Magazine]\n    ############################\n\n    Let's start, the question is: which tower is taller: eiffel tower or tower of pisa?\n\n    Thought:\n    thinking\n Action: google_search[What is taller, Eiffel Tower or Leaning Tower of Pisa]\n"
                                )
                            ],
                            _name=None,
                            _meta={},
                        )
                    ]
                },
                ("tool_extractor", 2): {
                    "messages": [
                        ChatMessage(
                            _role="user",
                            _content=[
                                TextContent(
                                    text="\n    Solve a question answering task with interleaving Thought, Action, Observation steps.\n\n    Thought reasons about the current situation\n\n    Action can be:\n    google_search - Searches Google for the exact concept/entity (given in square brackets) and returns the results for you to use\n    finish - Returns the final answer (given in square brackets) and finishes the task\n\n    Observation summarizes the Action outcome and helps in formulating the next\n    Thought in Thought, Action, Observation interleaving triplet of steps.\n\n    After each Observation, provide the next Thought and next Action.\n    Don't execute multiple steps even though you know the answer.\n    Only generate Thought and Action, never Observation, you'll get Observation from Action.\n    Follow the pattern in the example below.\n\n    Example:\n    ###########################\n    Question: Which magazine was started first Arthur’s Magazine or First for Women?\n    Thought: I need to search Arthur’s Magazine and First for Women, and find which was started\n    first.\n    Action: google_search[When was 'Arthur’s Magazine' started?]\n    Observation: Arthur’s Magazine was an American literary periodical ˘\n    published in Philadelphia and founded in 1844. Edited by Timothy Shay Arthur, it featured work by\n    Edgar A. Poe, J.H. Ingraham, Sarah Josepha Hale, Thomas G. Spear, and others. In May 1846\n    it was merged into Godey’s Lady’s Book.\n    Thought: Arthur’s Magazine was started in 1844. I need to search First for Women founding date next\n    Action: google_search[When was 'First for Women' magazine started?]\n    Observation: First for Women is a woman’s magazine published by Bauer Media Group in the\n    USA. The magazine was started in 1989. It is based in Englewood Cliffs, New Jersey. In 2011\n    the circulation of the magazine was 1,310,696 copies.\n    Thought: First for Women was started in 1989. 1844 (Arthur’s Magazine) ¡ 1989 (First for\n    Women), so Arthur’s Magazine was started first.\n    Action: finish[Arthur’s Magazine]\n    ############################\n\n    Let's start, the question is: which tower is taller: eiffel tower or tower of pisa?\n\n    Thought:\n    thinking\n Action: google_search[What is taller, Eiffel Tower or Leaning Tower of Pisa]\nObservation: Tower of Pisa is 55 meters tall\n\n\nThought: thinking\n Action: finish[Eiffel Tower]\n"
                                )
                            ],
                            _name=None,
                            _meta={},
                        )
                    ]
                },
            },
        )
    ]


@given("a pipeline that has a variadic component that receives partial inputs", target_fixture="pipeline_data")
def that_has_a_variadic_component_that_receives_partial_inputs(pipeline_class):
    @component
    class ConditionalDocumentCreator:
        def __init__(self, content: str):
            self._content = content

        @component.output_types(documents=List[Document], noop=None)
        def run(self, create_document: bool = False):
            if create_document:
                return {"documents": [Document(id=self._content, content=self._content)]}
            return {"noop": None}

    pipeline = pipeline_class(max_runs_per_component=1)
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
                expected_component_calls={
                    ("documents_joiner", 1): {
                        "documents": [
                            [Document(id="First document", content="First document")],
                            [Document(id="Third document", content="Third document")],
                        ],
                        "top_k": None,
                    },
                    ("first_creator", 1): {"create_document": True},
                    ("second_creator", 1): {"create_document": False},
                    ("third_creator", 1): {"create_document": True},
                },
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
                expected_component_calls={
                    ("documents_joiner", 1): {
                        "documents": [
                            [Document(id="First document", content="First document")],
                            [Document(id="Second document", content="Second document")],
                        ],
                        "top_k": None,
                    },
                    ("first_creator", 1): {"create_document": True},
                    ("second_creator", 1): {"create_document": True},
                    ("third_creator", 1): {"create_document": False},
                },
            ),
        ],
    )


@given(
    "a pipeline that has a variadic component that receives partial inputs in a different order",
    target_fixture="pipeline_data",
)
def that_has_a_variadic_component_that_receives_partial_inputs_different_order(pipeline_class):
    @component
    class ConditionalDocumentCreator:
        def __init__(self, content: str):
            self._content = content

        @component.output_types(documents=List[Document], noop=None)
        def run(self, create_document: bool = False):
            if create_document:
                return {"documents": [Document(id=self._content, content=self._content)]}
            return {"noop": None}

    pipeline = pipeline_class(max_runs_per_component=1)
    pipeline.add_component("third_creator", ConditionalDocumentCreator(content="Third document"))
    pipeline.add_component("first_creator", ConditionalDocumentCreator(content="First document"))
    pipeline.add_component("second_creator", ConditionalDocumentCreator(content="Second document"))
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
                expected_component_calls={
                    ("documents_joiner", 1): {
                        "documents": [
                            [Document(id="First document", content="First document")],
                            [Document(id="Third document", content="Third document")],
                        ],
                        "top_k": None,
                    },
                    ("first_creator", 1): {"create_document": True},
                    ("second_creator", 1): {"create_document": False},
                    ("third_creator", 1): {"create_document": True},
                },
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
                expected_component_calls={
                    ("documents_joiner", 1): {
                        "documents": [
                            [Document(id="First document", content="First document")],
                            [Document(id="Second document", content="Second document")],
                        ],
                        "top_k": None,
                    },
                    ("first_creator", 1): {"create_document": True},
                    ("second_creator", 1): {"create_document": True},
                    ("third_creator", 1): {"create_document": False},
                },
            ),
        ],
    )


@given("a pipeline that has an answer joiner variadic component", target_fixture="pipeline_data")
def that_has_an_answer_joiner_variadic_component(pipeline_class):
    query = "What's Natural Language Processing?"

    pipeline = pipeline_class(max_runs_per_component=1)
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
                expected_component_calls={
                    ("answer_builder_1", 1): {
                        "documents": None,
                        "meta": None,
                        "pattern": None,
                        "query": "What's Natural Language Processing?",
                        "reference_pattern": None,
                        "replies": ["This is a test answer"],
                    },
                    ("answer_builder_2", 1): {
                        "documents": None,
                        "meta": None,
                        "pattern": None,
                        "query": "What's Natural Language Processing?",
                        "reference_pattern": None,
                        "replies": ["This is a second test answer"],
                    },
                    ("answer_joiner", 1): {
                        "answers": [
                            [
                                GeneratedAnswer(
                                    data="This is a test answer",
                                    query="What's Natural Language Processing?",
                                    documents=[],
                                    meta={},
                                )
                            ],
                            [
                                GeneratedAnswer(
                                    data="This is a second test answer",
                                    query="What's Natural Language Processing?",
                                    documents=[],
                                    meta={},
                                )
                            ],
                        ],
                        "top_k": None,
                    },
                },
            )
        ],
    )


@given(
    "a pipeline that is linear and a component in the middle receives optional input from other components and input from the user",
    target_fixture="pipeline_data",
)
def that_is_linear_and_a_component_in_the_middle_receives_optional_input_from_other_components_and_input_from_the_user(
    pipeline_class,
):
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

    pipeline = pipeline_class()
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
                expected_component_calls={
                    ("builder", 1): {"template": None, "template_variables": None},
                    ("document_joiner", 1): {
                        "documents": [
                            [
                                Document(
                                    id="doc2",
                                    content="some text about investigation and treatment of Alzheimer disease",
                                    meta={"year": 2023, "disease": "Alzheimer", "author": "John Bread"},
                                    score=3.324112496100923,
                                )
                            ]
                        ],
                        "top_k": None,
                    },
                    ("metadata_extractor", 1): {"prompt": '{"disease": "Alzheimer", "year": 2023}'},
                    ("retriever", 1): {
                        "filters": {
                            "conditions": [
                                {"field": "meta.disease", "operator": "==", "value": "Alzheimer"},
                                {"field": "meta.year", "operator": "==", "value": 2023},
                            ],
                            "operator": "AND",
                        },
                        "query": "publications 2023 Alzheimer's disease",
                        "scale_score": None,
                        "top_k": None,
                    },
                },
            )
        ],
    )


@given("a pipeline that has a cycle that would get it stuck", target_fixture="pipeline_data")
def that_has_a_cycle_that_would_get_it_stuck(pipeline_class):
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

    pipeline = pipeline_class(max_runs_per_component=1)
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
def that_has_a_loop_in_the_middle(pipeline_class):
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

    pipeline = pipeline_class(max_runs_per_component=20)
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
                expected_component_calls={
                    ("answer_builder", 1): {
                        "documents": None,
                        "meta": None,
                        "pattern": None,
                        "query": "What is the answer?",
                        "reference_pattern": None,
                        "replies": ["42"],
                    },
                    ("answer_validator", 1): {"replies": ["No answer"]},
                    ("answer_validator", 2): {"replies": ["42"]},
                    ("llm", 1): {"prompt": "Random template"},
                    ("llm", 2): {"prompt": ""},
                    ("prompt_builder", 1): {
                        "invalid_replies": "",
                        "question": "What is the answer?",
                        "template": "Random template",
                        "template_variables": None,
                    },
                    ("prompt_builder", 2): {
                        "invalid_replies": ["No answer"],
                        "question": "What is the answer?",
                        "template": None,
                        "template_variables": None,
                    },
                    ("prompt_cleaner", 1): {"prompt": "Random template"},
                },
            )
        ],
    )


@given("a pipeline that has variadic component that receives a conditional input", target_fixture="pipeline_data")
def that_has_variadic_component_that_receives_a_conditional_input(pipeline_class):
    pipe = pipeline_class(max_runs_per_component=1)
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
            expected_component_calls={
                ("comma_splitter", 1): {
                    "documents": [
                        Document(
                            id="1000",
                            content="This document has so many, sentences. Like this one, or this one. Or even this other one.",
                        )
                    ]
                },
                ("conditional_router", 1): {
                    "documents": [
                        Document(
                            id="1000",
                            content="This document has so many, sentences. Like this one, or this one. Or even this other one.",
                        )
                    ]
                },
                ("document_cleaner", 1): {
                    "documents": [
                        Document(id="0", content="This document has so many"),
                        Document(id="1", content=" sentences. Like this one"),
                        Document(id="2", content=" or this one. Or even this other one."),
                    ]
                },
                ("document_joiner", 1): {
                    "documents": [
                        [
                            Document(id="0", content="This document has so many"),
                            Document(id="1", content=" sentences. Like this one"),
                            Document(id="2", content=" or this one. Or even this other one."),
                        ],
                        [
                            Document(id="0", content="This document has so many"),
                            Document(id="1", content="sentences. Like this one"),
                            Document(id="2", content="or this one. Or even this other one."),
                        ],
                    ],
                    "top_k": None,
                },
                ("noop2", 1): {
                    "documents": [
                        Document(
                            id="1000",
                            content="This document has so many, sentences. Like this one, or this one. Or even this other one.",
                        )
                    ]
                },
                ("noop3", 1): {
                    "documents": [
                        Document(
                            id="1000",
                            content="This document has so many, sentences. Like this one, or this one. Or even this other one.",
                        )
                    ]
                },
            },
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
            expected_component_calls={
                ("comma_splitter", 1): {
                    "documents": [
                        Document(
                            id="1000",
                            content="This document has so many, sentences. Like this one, or this one. Or even this other one.",
                        ),
                        Document(
                            id="1000",
                            content="This document has so many, sentences. Like this one, or this one. Or even this other one.",
                        ),
                    ]
                },
                ("conditional_router", 1): {
                    "documents": [
                        Document(
                            id="1000",
                            content="This document has so many, sentences. Like this one, or this one. Or even this other one.",
                        ),
                        Document(
                            id="1000",
                            content="This document has so many, sentences. Like this one, or this one. Or even this other one.",
                        ),
                    ]
                },
                ("document_cleaner", 1): {
                    "documents": [
                        Document(id="0", content="This document has so many"),
                        Document(id="1", content=" sentences. Like this one"),
                        Document(id="2", content=" or this one. Or even this other one."),
                        Document(id="3", content="This document has so many"),
                        Document(id="4", content=" sentences. Like this one"),
                        Document(id="5", content=" or this one. Or even this other one."),
                    ]
                },
                ("document_joiner", 1): {
                    "documents": [
                        [
                            Document(id="0", content="This document has so many"),
                            Document(id="1", content=" sentences. Like this one"),
                            Document(id="2", content=" or this one. Or even this other one."),
                            Document(id="3", content="This document has so many"),
                            Document(id="4", content=" sentences. Like this one"),
                            Document(id="5", content=" or this one. Or even this other one."),
                        ],
                        [
                            Document(id="0", content="This document has so many"),
                            Document(id="1", content="sentences. Like this one"),
                            Document(id="2", content="or this one. Or even this other one."),
                            Document(id="3", content="This document has so many"),
                            Document(id="4", content="sentences. Like this one"),
                            Document(id="5", content="or this one. Or even this other one."),
                        ],
                        [
                            Document(
                                id="1000",
                                content="This document has so many, sentences. Like this one, or this one. Or even this other one.",
                            ),
                            Document(
                                id="1000",
                                content="This document has so many, sentences. Like this one, or this one. Or even this other one.",
                            ),
                        ],
                    ],
                    "top_k": None,
                },
                ("empty_lines_cleaner", 1): {
                    "documents": [
                        Document(
                            id="1000",
                            content="This document has so many, sentences. Like this one, or this one. Or even this other one.",
                        ),
                        Document(
                            id="1000",
                            content="This document has so many, sentences. Like this one, or this one. Or even this other one.",
                        ),
                    ]
                },
                ("noop2", 1): {
                    "documents": [
                        Document(
                            id="1000",
                            content="This document has so many, sentences. Like this one, or this one. Or even this other one.",
                        ),
                        Document(
                            id="1000",
                            content="This document has so many, sentences. Like this one, or this one. Or even this other one.",
                        ),
                    ]
                },
                ("noop3", 1): {
                    "documents": [
                        Document(
                            id="1000",
                            content="This document has so many, sentences. Like this one, or this one. Or even this other one.",
                        ),
                        Document(
                            id="1000",
                            content="This document has so many, sentences. Like this one, or this one. Or even this other one.",
                        ),
                    ]
                },
            },
        ),
    ]


@given("a pipeline that has a string variadic component", target_fixture="pipeline_data")
def that_has_a_string_variadic_component(pipeline_class):
    string_1 = "What's Natural Language Processing?"
    string_2 = "What's is life?"

    pipeline = pipeline_class()
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
                expected_component_calls={
                    ("prompt_builder_1", 1): {
                        "query": "What's Natural Language Processing?",
                        "template": None,
                        "template_variables": None,
                    },
                    ("prompt_builder_2", 1): {"query": "What's is life?", "template": None, "template_variables": None},
                    ("string_joiner", 1): {
                        "strings": ["Builder 1: What's Natural Language Processing?", "Builder 2: What's is life?"]
                    },
                },
            )
        ],
    )


@given("a pipeline that is an agent that can use RAG", target_fixture="pipeline_data")
def an_agent_that_can_use_RAG(pipeline_class):
    @component
    class FixedGenerator:
        def __init__(self, replies):
            self.replies = replies
            self.idx = 0

        @component.output_types(replies=List[str])
        def run(self, prompt: str):
            if self.idx < len(self.replies):
                replies = [self.replies[self.idx]]
                self.idx += 1
            else:
                self.idx = 0
                replies = [self.replies[self.idx]]
                self.idx += 1

            return {"replies": replies}

    @component
    class FakeRetriever:
        @component.output_types(documents=List[Document])
        def run(self, query: str):
            return {
                "documents": [
                    Document(content="This is a document potentially answering the question.", meta={"access_group": 1})
                ]
            }

    agent_prompt_template = """
Your task is to answer the user's question.
You can use a RAG system to find information.
Use the RAG system until you have sufficient information to answer the question.
To use the RAG system, output "search:" followed by your question.
Once you have an answer, output "answer:" followed by your answer.

Here is the question: {{query}}
    """

    rag_prompt_template = """
Answer the question based on the provided documents.
Question: {{ query }}
Documents:
{% for document in documents %}
{{ document.content }}
{% endfor %}
    """

    joiner = BranchJoiner(type_=str)

    agent_llm = FixedGenerator(replies=["search: Can you help me?", "answer: here is my answer"])
    agent_prompt = PromptBuilder(template=agent_prompt_template)

    rag_llm = FixedGenerator(replies=["This is all the information I found!"])
    rag_prompt = PromptBuilder(template=rag_prompt_template)

    retriever = FakeRetriever()

    routes = [
        {
            "condition": "{{ 'search:' in replies[0] }}",
            "output": "{{ replies[0] }}",
            "output_name": "search",
            "output_type": str,
        },
        {
            "condition": "{{ 'answer:' in replies[0] }}",
            "output": "{{ replies }}",
            "output_name": "answer",
            "output_type": List[str],
        },
    ]

    router = ConditionalRouter(routes=routes)

    concatenator = OutputAdapter(template="{{current_prompt + '\n' + rag_answer[0]}}", output_type=str)

    answer_builder = AnswerBuilder()

    pp = pipeline_class(max_runs_per_component=2)

    pp.add_component("joiner", joiner)
    pp.add_component("rag_llm", rag_llm)
    pp.add_component("rag_prompt", rag_prompt)
    pp.add_component("agent_prompt", agent_prompt)
    pp.add_component("agent_llm", agent_llm)
    pp.add_component("router", router)
    pp.add_component("concatenator", concatenator)
    pp.add_component("retriever", retriever)
    pp.add_component("answer_builder", answer_builder)

    pp.connect("agent_prompt.prompt", "joiner.value")
    pp.connect("joiner.value", "agent_llm.prompt")
    pp.connect("agent_llm.replies", "router.replies")
    pp.connect("router.search", "retriever.query")
    pp.connect("router.answer", "answer_builder.replies")
    pp.connect("retriever.documents", "rag_prompt.documents")
    pp.connect("rag_prompt.prompt", "rag_llm.prompt")
    pp.connect("rag_llm.replies", "concatenator.rag_answer")
    pp.connect("joiner.value", "concatenator.current_prompt")
    pp.connect("concatenator.output", "joiner.value")

    query = "Does this run reliably?"

    return (
        pp,
        [
            PipelineRunData(
                inputs={
                    "agent_prompt": {"query": query},
                    "rag_prompt": {"query": query},
                    "answer_builder": {"query": query},
                },
                expected_outputs={
                    "answer_builder": {
                        "answers": [GeneratedAnswer(data="answer: here is my answer", query=query, documents=[])]
                    }
                },
                expected_component_calls={
                    ("agent_llm", 1): {
                        "prompt": "\n"
                        "Your task is to answer the user's question.\n"
                        "You can use a RAG system to find information.\n"
                        "Use the RAG system until you have sufficient "
                        "information to answer the question.\n"
                        'To use the RAG system, output "search:" '
                        "followed by your question.\n"
                        'Once you have an answer, output "answer:" '
                        "followed by your answer.\n"
                        "\n"
                        "Here is the question: Does this run reliably?\n"
                        "    "
                    },
                    ("agent_llm", 2): {
                        "prompt": "\n"
                        "Your task is to answer the user's question.\n"
                        "You can use a RAG system to find information.\n"
                        "Use the RAG system until you have sufficient "
                        "information to answer the question.\n"
                        'To use the RAG system, output "search:" '
                        "followed by your question.\n"
                        'Once you have an answer, output "answer:" '
                        "followed by your answer.\n"
                        "\n"
                        "Here is the question: Does this run reliably?\n"
                        "    \n"
                        "This is all the information I found!"
                    },
                    ("agent_prompt", 1): {
                        "query": "Does this run reliably?",
                        "template": None,
                        "template_variables": None,
                    },
                    ("answer_builder", 1): {
                        "documents": None,
                        "meta": None,
                        "pattern": None,
                        "query": "Does this run reliably?",
                        "reference_pattern": None,
                        "replies": ["answer: here is my answer"],
                    },
                    ("concatenator", 1): {
                        "current_prompt": "\n"
                        "Your task is to answer the user's "
                        "question.\n"
                        "You can use a RAG system to find "
                        "information.\n"
                        "Use the RAG system until you have "
                        "sufficient information to answer the "
                        "question.\n"
                        "To use the RAG system, output "
                        '"search:" followed by your '
                        "question.\n"
                        "Once you have an answer, output "
                        '"answer:" followed by your answer.\n'
                        "\n"
                        "Here is the question: Does this run "
                        "reliably?\n"
                        "    ",
                        "rag_answer": ["This is all the information I found!"],
                    },
                    ("joiner", 1): {
                        "value": [
                            "\n"
                            "Your task is to answer the user's question.\n"
                            "You can use a RAG system to find information.\n"
                            "Use the RAG system until you have sufficient "
                            "information to answer the question.\n"
                            'To use the RAG system, output "search:" followed '
                            "by your question.\n"
                            'Once you have an answer, output "answer:" followed '
                            "by your answer.\n"
                            "\n"
                            "Here is the question: Does this run reliably?\n"
                            "    "
                        ]
                    },
                    ("joiner", 2): {
                        "value": [
                            "\n"
                            "Your task is to answer the user's question.\n"
                            "You can use a RAG system to find information.\n"
                            "Use the RAG system until you have sufficient "
                            "information to answer the question.\n"
                            'To use the RAG system, output "search:" followed '
                            "by your question.\n"
                            'Once you have an answer, output "answer:" followed '
                            "by your answer.\n"
                            "\n"
                            "Here is the question: Does this run reliably?\n"
                            "    \n"
                            "This is all the information I found!"
                        ]
                    },
                    ("rag_llm", 1): {
                        "prompt": "\n"
                        "Answer the question based on the provided "
                        "documents.\n"
                        "Question: Does this run reliably?\n"
                        "Documents:\n"
                        "\n"
                        "This is a document potentially answering the "
                        "question.\n"
                        "\n"
                        "    "
                    },
                    ("rag_prompt", 1): {
                        "documents": [
                            Document(
                                id="969664d0cf76e52b0ffb719d00d3e5a6b1c90bb29e56f6107dfd87bf2f5388ed",
                                content="This is a document potentially answering the question.",
                                meta={"access_group": 1},
                            )
                        ],
                        "query": "Does this run reliably?",
                        "template": None,
                        "template_variables": None,
                    },
                    ("retriever", 1): {"query": "search: Can you help me?"},
                    ("router", 1): {"replies": ["search: Can you help me?"]},
                    ("router", 2): {"replies": ["answer: here is my answer"]},
                },
            )
        ],
    )


@given("a pipeline that has a feedback loop", target_fixture="pipeline_data")
def has_feedback_loop(pipeline_class):
    @component
    class FixedGenerator:
        def __init__(self, replies):
            self.replies = replies
            self.idx = 0

        @component.output_types(replies=List[str])
        def run(self, prompt: str):
            if self.idx < len(self.replies):
                replies = [self.replies[self.idx]]
                self.idx += 1
            else:
                self.idx = 0
                replies = [self.replies[self.idx]]
                self.idx += 1

            return {"replies": replies}

    code_prompt_template = """
Generate code to solve the task: {{ task }}

{% if feedback %}
Here is your initial attempt and some feedback:
{{ feedback }}
{% endif %}
    """

    feedback_prompt_template = """
Check if this code is valid and can run: {{ code[0] }}
Return "PASS" if it passes and "FAIL" if it fails.
Provide additional feedback on why it fails.
    """

    code_llm = FixedGenerator(replies=["invalid code", "valid code"])
    code_prompt = PromptBuilder(template=code_prompt_template)

    feedback_llm = FixedGenerator(replies=["FAIL", "PASS"])
    feedback_prompt = PromptBuilder(template=feedback_prompt_template)

    routes = [
        {
            "condition": "{{ 'FAIL' in replies[0] }}",
            "output": "{{ replies[0] }}",
            "output_name": "fail",
            "output_type": str,
        },
        {
            "condition": "{{ 'PASS' in replies[0] }}",
            "output": "{{ code }}",
            "output_name": "pass",
            "output_type": List[str],
        },
    ]

    router = ConditionalRouter(routes=routes)

    concatenator = OutputAdapter(template="{{current_prompt[0] + '\n' + feedback[0]}}", output_type=str)

    answer_builder = AnswerBuilder()

    pp = pipeline_class(max_runs_per_component=100)

    pp.add_component("code_llm", code_llm)
    pp.add_component("code_prompt", code_prompt)
    pp.add_component("feedback_prompt", feedback_prompt)
    pp.add_component("feedback_llm", feedback_llm)
    pp.add_component("router", router)
    pp.add_component("concatenator", concatenator)
    pp.add_component("answer_builder", answer_builder)

    pp.connect("code_prompt.prompt", "code_llm.prompt")
    pp.connect("code_llm.replies", "feedback_prompt.code")
    pp.connect("feedback_llm.replies", "router.replies")
    pp.connect("router.fail", "concatenator.feedback")
    pp.connect("router.pass", "answer_builder.replies")
    pp.connect("code_llm.replies", "router.code")
    pp.connect("feedback_prompt.prompt", "feedback_llm.prompt")
    pp.connect("code_llm.replies", "concatenator.current_prompt")
    pp.connect("concatenator.output", "code_prompt.feedback")

    task = "Generate code to generate christmas ascii-art"

    return (
        pp,
        [
            PipelineRunData(
                inputs={"code_prompt": {"task": task}, "answer_builder": {"query": task}},
                expected_outputs={
                    "answer_builder": {"answers": [GeneratedAnswer(data="valid code", query=task, documents=[])]}
                },
                expected_component_calls={
                    ("answer_builder", 1): {
                        "documents": None,
                        "meta": None,
                        "pattern": None,
                        "query": "Generate code to generate christmas ascii-art",
                        "reference_pattern": None,
                        "replies": ["valid code"],
                    },
                    ("code_llm", 1): {
                        "prompt": "\n"
                        "Generate code to solve the task: Generate code "
                        "to generate christmas ascii-art\n"
                        "\n"
                        "\n"
                        "    "
                    },
                    ("code_llm", 2): {
                        "prompt": "\n"
                        "Generate code to solve the task: Generate code "
                        "to generate christmas ascii-art\n"
                        "\n"
                        "\n"
                        "Here is your initial attempt and some feedback:\n"
                        "invalid code\n"
                        "F\n"
                        "\n"
                        "    "
                    },
                    ("code_prompt", 1): {
                        "feedback": "",
                        "task": "Generate code to generate christmas ascii-art",
                        "template": None,
                        "template_variables": None,
                    },
                    ("code_prompt", 2): {
                        "feedback": "invalid code\nF",
                        "task": "Generate code to generate christmas ascii-art",
                        "template": None,
                        "template_variables": None,
                    },
                    ("concatenator", 1): {"current_prompt": ["invalid code"], "feedback": "FAIL"},
                    ("feedback_llm", 1): {
                        "prompt": "\n"
                        "Check if this code is valid and can run: "
                        "invalid code\n"
                        'Return "PASS" if it passes and "FAIL" if it '
                        "fails.\n"
                        "Provide additional feedback on why it "
                        "fails.\n"
                        "    "
                    },
                    ("feedback_llm", 2): {
                        "prompt": "\n"
                        "Check if this code is valid and can run: "
                        "valid code\n"
                        'Return "PASS" if it passes and "FAIL" if it '
                        "fails.\n"
                        "Provide additional feedback on why it "
                        "fails.\n"
                        "    "
                    },
                    ("feedback_prompt", 1): {"code": ["invalid code"], "template": None, "template_variables": None},
                    ("feedback_prompt", 2): {"code": ["valid code"], "template": None, "template_variables": None},
                    ("router", 1): {"code": ["invalid code"], "replies": ["FAIL"]},
                    ("router", 2): {"code": ["valid code"], "replies": ["PASS"]},
                },
            )
        ],
    )


@given("a pipeline created in a non-standard order that has a loop", target_fixture="pipeline_data")
def has_non_standard_order_loop(pipeline_class):
    @component
    class FixedGenerator:
        def __init__(self, replies):
            self.replies = replies
            self.idx = 0

        @component.output_types(replies=List[str])
        def run(self, prompt: str):
            if self.idx < len(self.replies):
                replies = [self.replies[self.idx]]
                self.idx += 1
            else:
                self.idx = 0
                replies = [self.replies[self.idx]]
                self.idx += 1

            return {"replies": replies}

    code_prompt_template = """
Generate code to solve the task: {{ task }}

{% if feedback %}
Here is your initial attempt and some feedback:
{{ feedback }}
{% endif %}
    """

    feedback_prompt_template = """
Check if this code is valid and can run: {{ code[0] }}
Return "PASS" if it passes and "FAIL" if it fails.
Provide additional feedback on why it fails.
    """

    code_llm = FixedGenerator(replies=["invalid code", "valid code"])
    code_prompt = PromptBuilder(template=code_prompt_template)

    feedback_llm = FixedGenerator(replies=["FAIL", "PASS"])
    feedback_prompt = PromptBuilder(template=feedback_prompt_template)

    routes = [
        {
            "condition": "{{ 'FAIL' in replies[0] }}",
            "output": "{{ replies[0] }}",
            "output_name": "fail",
            "output_type": str,
        },
        {
            "condition": "{{ 'PASS' in replies[0] }}",
            "output": "{{ code }}",
            "output_name": "pass",
            "output_type": List[str],
        },
    ]

    router = ConditionalRouter(routes=routes)

    concatenator = OutputAdapter(template="{{current_prompt[0] + '\n' + feedback[0]}}", output_type=str)

    answer_builder = AnswerBuilder()

    pp = pipeline_class(max_runs_per_component=100)

    pp.add_component("concatenator", concatenator)
    pp.add_component("code_llm", code_llm)
    pp.add_component("code_prompt", code_prompt)
    pp.add_component("feedback_prompt", feedback_prompt)
    pp.add_component("feedback_llm", feedback_llm)
    pp.add_component("router", router)

    pp.add_component("answer_builder", answer_builder)

    pp.connect("concatenator.output", "code_prompt.feedback")
    pp.connect("code_prompt.prompt", "code_llm.prompt")
    pp.connect("code_llm.replies", "feedback_prompt.code")
    pp.connect("feedback_llm.replies", "router.replies")
    pp.connect("router.fail", "concatenator.feedback")
    pp.connect("feedback_prompt.prompt", "feedback_llm.prompt")
    pp.connect("router.pass", "answer_builder.replies")
    pp.connect("code_llm.replies", "router.code")
    pp.connect("code_llm.replies", "concatenator.current_prompt")

    task = "Generate code to generate christmas ascii-art"

    return (
        pp,
        [
            PipelineRunData(
                inputs={"code_prompt": {"task": task}, "answer_builder": {"query": task}},
                expected_outputs={
                    "answer_builder": {"answers": [GeneratedAnswer(data="valid code", query=task, documents=[])]}
                },
                expected_component_calls={
                    ("answer_builder", 1): {
                        "documents": None,
                        "meta": None,
                        "pattern": None,
                        "query": "Generate code to generate christmas ascii-art",
                        "reference_pattern": None,
                        "replies": ["valid code"],
                    },
                    ("code_llm", 1): {
                        "prompt": "\n"
                        "Generate code to solve the task: Generate code "
                        "to generate christmas ascii-art\n"
                        "\n"
                        "\n"
                        "    "
                    },
                    ("code_llm", 2): {
                        "prompt": "\n"
                        "Generate code to solve the task: Generate code "
                        "to generate christmas ascii-art\n"
                        "\n"
                        "\n"
                        "Here is your initial attempt and some feedback:\n"
                        "invalid code\n"
                        "F\n"
                        "\n"
                        "    "
                    },
                    ("code_prompt", 1): {
                        "feedback": "",
                        "task": "Generate code to generate christmas ascii-art",
                        "template": None,
                        "template_variables": None,
                    },
                    ("code_prompt", 2): {
                        "feedback": "invalid code\nF",
                        "task": "Generate code to generate christmas ascii-art",
                        "template": None,
                        "template_variables": None,
                    },
                    ("concatenator", 1): {"current_prompt": ["invalid code"], "feedback": "FAIL"},
                    ("feedback_llm", 1): {
                        "prompt": "\n"
                        "Check if this code is valid and can run: "
                        "invalid code\n"
                        'Return "PASS" if it passes and "FAIL" if it '
                        "fails.\n"
                        "Provide additional feedback on why it "
                        "fails.\n"
                        "    "
                    },
                    ("feedback_llm", 2): {
                        "prompt": "\n"
                        "Check if this code is valid and can run: "
                        "valid code\n"
                        'Return "PASS" if it passes and "FAIL" if it '
                        "fails.\n"
                        "Provide additional feedback on why it "
                        "fails.\n"
                        "    "
                    },
                    ("feedback_prompt", 1): {"code": ["invalid code"], "template": None, "template_variables": None},
                    ("feedback_prompt", 2): {"code": ["valid code"], "template": None, "template_variables": None},
                    ("router", 1): {"code": ["invalid code"], "replies": ["FAIL"]},
                    ("router", 2): {"code": ["valid code"], "replies": ["PASS"]},
                },
            )
        ],
    )


@given("a pipeline that has an agent with a feedback cycle", target_fixture="pipeline_data")
def agent_with_feedback_cycle(pipeline_class):
    @component
    class FixedGenerator:
        def __init__(self, replies):
            self.replies = replies
            self.idx = 0

        @component.output_types(replies=List[str])
        def run(self, prompt: str):
            if self.idx < len(self.replies):
                replies = [self.replies[self.idx]]
                self.idx += 1
            else:
                self.idx = 0
                replies = [self.replies[self.idx]]
                self.idx += 1

            return {"replies": replies}

    @component
    class FakeFileEditor:
        @component.output_types(files=str)
        def run(self, replies: List[str]):
            return {"files": "This is the edited file content."}

    code_prompt_template = """
Generate code to solve the task: {{ task }}

You can edit files by returning:
Edit: file_name

Once you solved the task, respond with:
Task finished!

{% if feedback %}
Here is your initial attempt and some feedback:
{{ feedback }}
{% endif %}
    """

    feedback_prompt_template = """
{% if task_finished %}
Check if this code is valid and can run: {{ code }}
Return "PASS" if it passes and "FAIL" if it fails.
Provide additional feedback on why it fails.
{% endif %}
    """

    code_llm = FixedGenerator(replies=["Edit: file_1.py", "Edit: file_2.py", "Edit: file_3.py", "Task finished!"])
    code_prompt = PromptBuilder(template=code_prompt_template)
    file_editor = FakeFileEditor()

    feedback_llm = FixedGenerator(replies=["FAIL", "PASS"])
    feedback_prompt = PromptBuilder(template=feedback_prompt_template, required_variables=["task_finished"])

    routes = [
        {
            "condition": "{{ 'FAIL' in replies[0] }}",
            "output": "{{ current_prompt + '\n' + replies[0] }}",
            "output_name": "fail",
            "output_type": str,
        },
        {
            "condition": "{{ 'PASS' in replies[0] }}",
            "output": "{{ replies }}",
            "output_name": "pass",
            "output_type": List[str],
        },
    ]
    feedback_router = ConditionalRouter(routes=routes)

    tool_use_routes = [
        {
            "condition": "{{ 'Edit:' in replies[0] }}",
            "output": "{{ replies }}",
            "output_name": "edit",
            "output_type": List[str],
        },
        {
            "condition": "{{ 'Task finished!' in replies[0] }}",
            "output": "{{ replies }}",
            "output_name": "done",
            "output_type": List[str],
        },
    ]
    tool_use_router = ConditionalRouter(routes=tool_use_routes)

    joiner = BranchJoiner(type_=str)
    agent_concatenator = OutputAdapter(template="{{current_prompt + '\n' + files}}", output_type=str)

    pp = pipeline_class(max_runs_per_component=100)

    pp.add_component("code_prompt", code_prompt)
    pp.add_component("joiner", joiner)
    pp.add_component("code_llm", code_llm)
    pp.add_component("tool_use_router", tool_use_router)
    pp.add_component("file_editor", file_editor)
    pp.add_component("agent_concatenator", agent_concatenator)
    pp.add_component("feedback_prompt", feedback_prompt)
    pp.add_component("feedback_llm", feedback_llm)
    pp.add_component("feedback_router", feedback_router)

    # Main Agent
    pp.connect("code_prompt.prompt", "joiner.value")
    pp.connect("joiner.value", "code_llm.prompt")
    pp.connect("code_llm.replies", "tool_use_router.replies")
    pp.connect("tool_use_router.edit", "file_editor.replies")
    pp.connect("file_editor.files", "agent_concatenator.files")
    pp.connect("joiner.value", "agent_concatenator.current_prompt")
    pp.connect("agent_concatenator.output", "joiner.value")

    # Feedback Cycle
    pp.connect("tool_use_router.done", "feedback_prompt.task_finished")
    pp.connect("agent_concatenator.output", "feedback_prompt.code")
    pp.connect("feedback_prompt.prompt", "feedback_llm.prompt")
    pp.connect("feedback_llm.replies", "feedback_router.replies")
    pp.connect("agent_concatenator.output", "feedback_router.current_prompt")
    pp.connect("feedback_router.fail", "joiner.value")

    task = "Generate code to generate christmas ascii-art"

    return (
        pp,
        [
            PipelineRunData(
                inputs={"code_prompt": {"task": task}},
                expected_outputs={"feedback_router": {"pass": ["PASS"]}},
                expected_component_calls={
                    ("agent_concatenator", 1): {
                        "current_prompt": "\n"
                        "Generate code to solve the "
                        "task: Generate code to "
                        "generate christmas ascii-art\n"
                        "\n"
                        "You can edit files by "
                        "returning:\n"
                        "Edit: file_name\n"
                        "\n"
                        "Once you solved the task, "
                        "respond with:\n"
                        "Task finished!\n"
                        "\n"
                        "\n"
                        "    ",
                        "files": "This is the edited file content.",
                    },
                    ("agent_concatenator", 2): {
                        "current_prompt": "\n"
                        "Generate code to solve the "
                        "task: Generate code to "
                        "generate christmas ascii-art\n"
                        "\n"
                        "You can edit files by "
                        "returning:\n"
                        "Edit: file_name\n"
                        "\n"
                        "Once you solved the task, "
                        "respond with:\n"
                        "Task finished!\n"
                        "\n"
                        "\n"
                        "    \n"
                        "This is the edited file "
                        "content.",
                        "files": "This is the edited file content.",
                    },
                    ("agent_concatenator", 3): {
                        "current_prompt": "\n"
                        "Generate code to solve the "
                        "task: Generate code to "
                        "generate christmas ascii-art\n"
                        "\n"
                        "You can edit files by "
                        "returning:\n"
                        "Edit: file_name\n"
                        "\n"
                        "Once you solved the task, "
                        "respond with:\n"
                        "Task finished!\n"
                        "\n"
                        "\n"
                        "    \n"
                        "This is the edited file "
                        "content.\n"
                        "This is the edited file "
                        "content.",
                        "files": "This is the edited file content.",
                    },
                    ("agent_concatenator", 4): {
                        "current_prompt": "\n"
                        "Generate code to solve the "
                        "task: Generate code to "
                        "generate christmas ascii-art\n"
                        "\n"
                        "You can edit files by "
                        "returning:\n"
                        "Edit: file_name\n"
                        "\n"
                        "Once you solved the task, "
                        "respond with:\n"
                        "Task finished!\n"
                        "\n"
                        "\n"
                        "    \n"
                        "This is the edited file "
                        "content.\n"
                        "This is the edited file "
                        "content.\n"
                        "This is the edited file "
                        "content.\n"
                        "FAIL",
                        "files": "This is the edited file content.",
                    },
                    ("agent_concatenator", 5): {
                        "current_prompt": "\n"
                        "Generate code to solve the "
                        "task: Generate code to "
                        "generate christmas ascii-art\n"
                        "\n"
                        "You can edit files by "
                        "returning:\n"
                        "Edit: file_name\n"
                        "\n"
                        "Once you solved the task, "
                        "respond with:\n"
                        "Task finished!\n"
                        "\n"
                        "\n"
                        "    \n"
                        "This is the edited file "
                        "content.\n"
                        "This is the edited file "
                        "content.\n"
                        "This is the edited file "
                        "content.\n"
                        "FAIL\n"
                        "This is the edited file "
                        "content.",
                        "files": "This is the edited file content.",
                    },
                    ("agent_concatenator", 6): {
                        "current_prompt": "\n"
                        "Generate code to solve the "
                        "task: Generate code to "
                        "generate christmas ascii-art\n"
                        "\n"
                        "You can edit files by "
                        "returning:\n"
                        "Edit: file_name\n"
                        "\n"
                        "Once you solved the task, "
                        "respond with:\n"
                        "Task finished!\n"
                        "\n"
                        "\n"
                        "    \n"
                        "This is the edited file "
                        "content.\n"
                        "This is the edited file "
                        "content.\n"
                        "This is the edited file "
                        "content.\n"
                        "FAIL\n"
                        "This is the edited file "
                        "content.\n"
                        "This is the edited file "
                        "content.",
                        "files": "This is the edited file content.",
                    },
                    ("code_llm", 1): {
                        "prompt": "\n"
                        "Generate code to solve the task: Generate code "
                        "to generate christmas ascii-art\n"
                        "\n"
                        "You can edit files by returning:\n"
                        "Edit: file_name\n"
                        "\n"
                        "Once you solved the task, respond with:\n"
                        "Task finished!\n"
                        "\n"
                        "\n"
                        "    "
                    },
                    ("code_llm", 2): {
                        "prompt": "\n"
                        "Generate code to solve the task: Generate code "
                        "to generate christmas ascii-art\n"
                        "\n"
                        "You can edit files by returning:\n"
                        "Edit: file_name\n"
                        "\n"
                        "Once you solved the task, respond with:\n"
                        "Task finished!\n"
                        "\n"
                        "\n"
                        "    \n"
                        "This is the edited file content."
                    },
                    ("code_llm", 3): {
                        "prompt": "\n"
                        "Generate code to solve the task: Generate code "
                        "to generate christmas ascii-art\n"
                        "\n"
                        "You can edit files by returning:\n"
                        "Edit: file_name\n"
                        "\n"
                        "Once you solved the task, respond with:\n"
                        "Task finished!\n"
                        "\n"
                        "\n"
                        "    \n"
                        "This is the edited file content.\n"
                        "This is the edited file content."
                    },
                    ("code_llm", 4): {
                        "prompt": "\n"
                        "Generate code to solve the task: Generate code "
                        "to generate christmas ascii-art\n"
                        "\n"
                        "You can edit files by returning:\n"
                        "Edit: file_name\n"
                        "\n"
                        "Once you solved the task, respond with:\n"
                        "Task finished!\n"
                        "\n"
                        "\n"
                        "    \n"
                        "This is the edited file content.\n"
                        "This is the edited file content.\n"
                        "This is the edited file content."
                    },
                    ("code_llm", 5): {
                        "prompt": "\n"
                        "Generate code to solve the task: Generate code "
                        "to generate christmas ascii-art\n"
                        "\n"
                        "You can edit files by returning:\n"
                        "Edit: file_name\n"
                        "\n"
                        "Once you solved the task, respond with:\n"
                        "Task finished!\n"
                        "\n"
                        "\n"
                        "    \n"
                        "This is the edited file content.\n"
                        "This is the edited file content.\n"
                        "This is the edited file content.\n"
                        "FAIL"
                    },
                    ("code_llm", 6): {
                        "prompt": "\n"
                        "Generate code to solve the task: Generate code "
                        "to generate christmas ascii-art\n"
                        "\n"
                        "You can edit files by returning:\n"
                        "Edit: file_name\n"
                        "\n"
                        "Once you solved the task, respond with:\n"
                        "Task finished!\n"
                        "\n"
                        "\n"
                        "    \n"
                        "This is the edited file content.\n"
                        "This is the edited file content.\n"
                        "This is the edited file content.\n"
                        "FAIL\n"
                        "This is the edited file content."
                    },
                    ("code_llm", 7): {
                        "prompt": "\n"
                        "Generate code to solve the task: Generate code "
                        "to generate christmas ascii-art\n"
                        "\n"
                        "You can edit files by returning:\n"
                        "Edit: file_name\n"
                        "\n"
                        "Once you solved the task, respond with:\n"
                        "Task finished!\n"
                        "\n"
                        "\n"
                        "    \n"
                        "This is the edited file content.\n"
                        "This is the edited file content.\n"
                        "This is the edited file content.\n"
                        "FAIL\n"
                        "This is the edited file content.\n"
                        "This is the edited file content."
                    },
                    ("code_llm", 8): {
                        "prompt": "\n"
                        "Generate code to solve the task: Generate code "
                        "to generate christmas ascii-art\n"
                        "\n"
                        "You can edit files by returning:\n"
                        "Edit: file_name\n"
                        "\n"
                        "Once you solved the task, respond with:\n"
                        "Task finished!\n"
                        "\n"
                        "\n"
                        "    \n"
                        "This is the edited file content.\n"
                        "This is the edited file content.\n"
                        "This is the edited file content.\n"
                        "FAIL\n"
                        "This is the edited file content.\n"
                        "This is the edited file content.\n"
                        "This is the edited file content."
                    },
                    ("code_prompt", 1): {
                        "feedback": "",
                        "task": "Generate code to generate christmas ascii-art",
                        "template": None,
                        "template_variables": None,
                    },
                    ("feedback_llm", 1): {
                        "prompt": "\n"
                        "\n"
                        "Check if this code is valid and can run: \n"
                        "Generate code to solve the task: Generate "
                        "code to generate christmas ascii-art\n"
                        "\n"
                        "You can edit files by returning:\n"
                        "Edit: file_name\n"
                        "\n"
                        "Once you solved the task, respond with:\n"
                        "Task finished!\n"
                        "\n"
                        "\n"
                        "    \n"
                        "This is the edited file content.\n"
                        "This is the edited file content.\n"
                        "This is the edited file content.\n"
                        'Return "PASS" if it passes and "FAIL" if it '
                        "fails.\n"
                        "Provide additional feedback on why it "
                        "fails.\n"
                        "\n"
                        "    "
                    },
                    ("feedback_llm", 2): {
                        "prompt": "\n"
                        "\n"
                        "Check if this code is valid and can run: \n"
                        "Generate code to solve the task: Generate "
                        "code to generate christmas ascii-art\n"
                        "\n"
                        "You can edit files by returning:\n"
                        "Edit: file_name\n"
                        "\n"
                        "Once you solved the task, respond with:\n"
                        "Task finished!\n"
                        "\n"
                        "\n"
                        "    \n"
                        "This is the edited file content.\n"
                        "This is the edited file content.\n"
                        "This is the edited file content.\n"
                        "FAIL\n"
                        "This is the edited file content.\n"
                        "This is the edited file content.\n"
                        "This is the edited file content.\n"
                        'Return "PASS" if it passes and "FAIL" if it '
                        "fails.\n"
                        "Provide additional feedback on why it "
                        "fails.\n"
                        "\n"
                        "    "
                    },
                    ("feedback_prompt", 1): {
                        "code": "\n"
                        "Generate code to solve the task: Generate "
                        "code to generate christmas ascii-art\n"
                        "\n"
                        "You can edit files by returning:\n"
                        "Edit: file_name\n"
                        "\n"
                        "Once you solved the task, respond with:\n"
                        "Task finished!\n"
                        "\n"
                        "\n"
                        "    \n"
                        "This is the edited file content.\n"
                        "This is the edited file content.\n"
                        "This is the edited file content.",
                        "task_finished": ["Task finished!"],
                        "template": None,
                        "template_variables": None,
                    },
                    ("feedback_prompt", 2): {
                        "code": "\n"
                        "Generate code to solve the task: Generate "
                        "code to generate christmas ascii-art\n"
                        "\n"
                        "You can edit files by returning:\n"
                        "Edit: file_name\n"
                        "\n"
                        "Once you solved the task, respond with:\n"
                        "Task finished!\n"
                        "\n"
                        "\n"
                        "    \n"
                        "This is the edited file content.\n"
                        "This is the edited file content.\n"
                        "This is the edited file content.\n"
                        "FAIL\n"
                        "This is the edited file content.\n"
                        "This is the edited file content.\n"
                        "This is the edited file content.",
                        "task_finished": ["Task finished!"],
                        "template": None,
                        "template_variables": None,
                    },
                    ("feedback_router", 1): {
                        "current_prompt": "\n"
                        "Generate code to solve the task: "
                        "Generate code to generate "
                        "christmas ascii-art\n"
                        "\n"
                        "You can edit files by returning:\n"
                        "Edit: file_name\n"
                        "\n"
                        "Once you solved the task, respond "
                        "with:\n"
                        "Task finished!\n"
                        "\n"
                        "\n"
                        "    \n"
                        "This is the edited file content.\n"
                        "This is the edited file content.\n"
                        "This is the edited file content.",
                        "replies": ["FAIL"],
                    },
                    ("feedback_router", 2): {
                        "current_prompt": "\n"
                        "Generate code to solve the task: "
                        "Generate code to generate "
                        "christmas ascii-art\n"
                        "\n"
                        "You can edit files by returning:\n"
                        "Edit: file_name\n"
                        "\n"
                        "Once you solved the task, respond "
                        "with:\n"
                        "Task finished!\n"
                        "\n"
                        "\n"
                        "    \n"
                        "This is the edited file content.\n"
                        "This is the edited file content.\n"
                        "This is the edited file content.\n"
                        "FAIL\n"
                        "This is the edited file content.\n"
                        "This is the edited file content.\n"
                        "This is the edited file content.",
                        "replies": ["PASS"],
                    },
                    ("file_editor", 1): {"replies": ["Edit: file_1.py"]},
                    ("file_editor", 2): {"replies": ["Edit: file_2.py"]},
                    ("file_editor", 3): {"replies": ["Edit: file_3.py"]},
                    ("file_editor", 4): {"replies": ["Edit: file_1.py"]},
                    ("file_editor", 5): {"replies": ["Edit: file_2.py"]},
                    ("file_editor", 6): {"replies": ["Edit: file_3.py"]},
                    ("joiner", 1): {
                        "value": [
                            "\n"
                            "Generate code to solve the task: Generate code to "
                            "generate christmas ascii-art\n"
                            "\n"
                            "You can edit files by returning:\n"
                            "Edit: file_name\n"
                            "\n"
                            "Once you solved the task, respond with:\n"
                            "Task finished!\n"
                            "\n"
                            "\n"
                            "    "
                        ]
                    },
                    ("joiner", 2): {
                        "value": [
                            "\n"
                            "Generate code to solve the task: Generate code to "
                            "generate christmas ascii-art\n"
                            "\n"
                            "You can edit files by returning:\n"
                            "Edit: file_name\n"
                            "\n"
                            "Once you solved the task, respond with:\n"
                            "Task finished!\n"
                            "\n"
                            "\n"
                            "    \n"
                            "This is the edited file content."
                        ]
                    },
                    ("joiner", 3): {
                        "value": [
                            "\n"
                            "Generate code to solve the task: Generate code to "
                            "generate christmas ascii-art\n"
                            "\n"
                            "You can edit files by returning:\n"
                            "Edit: file_name\n"
                            "\n"
                            "Once you solved the task, respond with:\n"
                            "Task finished!\n"
                            "\n"
                            "\n"
                            "    \n"
                            "This is the edited file content.\n"
                            "This is the edited file content."
                        ]
                    },
                    ("joiner", 4): {
                        "value": [
                            "\n"
                            "Generate code to solve the task: Generate code to "
                            "generate christmas ascii-art\n"
                            "\n"
                            "You can edit files by returning:\n"
                            "Edit: file_name\n"
                            "\n"
                            "Once you solved the task, respond with:\n"
                            "Task finished!\n"
                            "\n"
                            "\n"
                            "    \n"
                            "This is the edited file content.\n"
                            "This is the edited file content.\n"
                            "This is the edited file content."
                        ]
                    },
                    ("joiner", 5): {
                        "value": [
                            "\n"
                            "Generate code to solve the task: Generate code to "
                            "generate christmas ascii-art\n"
                            "\n"
                            "You can edit files by returning:\n"
                            "Edit: file_name\n"
                            "\n"
                            "Once you solved the task, respond with:\n"
                            "Task finished!\n"
                            "\n"
                            "\n"
                            "    \n"
                            "This is the edited file content.\n"
                            "This is the edited file content.\n"
                            "This is the edited file content.\n"
                            "FAIL"
                        ]
                    },
                    ("joiner", 6): {
                        "value": [
                            "\n"
                            "Generate code to solve the task: Generate code to "
                            "generate christmas ascii-art\n"
                            "\n"
                            "You can edit files by returning:\n"
                            "Edit: file_name\n"
                            "\n"
                            "Once you solved the task, respond with:\n"
                            "Task finished!\n"
                            "\n"
                            "\n"
                            "    \n"
                            "This is the edited file content.\n"
                            "This is the edited file content.\n"
                            "This is the edited file content.\n"
                            "FAIL\n"
                            "This is the edited file content."
                        ]
                    },
                    ("joiner", 7): {
                        "value": [
                            "\n"
                            "Generate code to solve the task: Generate code to "
                            "generate christmas ascii-art\n"
                            "\n"
                            "You can edit files by returning:\n"
                            "Edit: file_name\n"
                            "\n"
                            "Once you solved the task, respond with:\n"
                            "Task finished!\n"
                            "\n"
                            "\n"
                            "    \n"
                            "This is the edited file content.\n"
                            "This is the edited file content.\n"
                            "This is the edited file content.\n"
                            "FAIL\n"
                            "This is the edited file content.\n"
                            "This is the edited file content."
                        ]
                    },
                    ("joiner", 8): {
                        "value": [
                            "\n"
                            "Generate code to solve the task: Generate code to "
                            "generate christmas ascii-art\n"
                            "\n"
                            "You can edit files by returning:\n"
                            "Edit: file_name\n"
                            "\n"
                            "Once you solved the task, respond with:\n"
                            "Task finished!\n"
                            "\n"
                            "\n"
                            "    \n"
                            "This is the edited file content.\n"
                            "This is the edited file content.\n"
                            "This is the edited file content.\n"
                            "FAIL\n"
                            "This is the edited file content.\n"
                            "This is the edited file content.\n"
                            "This is the edited file content."
                        ]
                    },
                    ("tool_use_router", 1): {"replies": ["Edit: file_1.py"]},
                    ("tool_use_router", 2): {"replies": ["Edit: file_2.py"]},
                    ("tool_use_router", 3): {"replies": ["Edit: file_3.py"]},
                    ("tool_use_router", 4): {"replies": ["Task finished!"]},
                    ("tool_use_router", 5): {"replies": ["Edit: file_1.py"]},
                    ("tool_use_router", 6): {"replies": ["Edit: file_2.py"]},
                    ("tool_use_router", 7): {"replies": ["Edit: file_3.py"]},
                    ("tool_use_router", 8): {"replies": ["Task finished!"]},
                },
            )
        ],
    )


@given("a pipeline that passes outputs that are consumed in cycle to outside the cycle", target_fixture="pipeline_data")
def passes_outputs_outside_cycle(pipeline_class):
    @component
    class FixedGenerator:
        def __init__(self, replies):
            self.replies = replies
            self.idx = 0

        @component.output_types(replies=List[str])
        def run(self, prompt: str):
            if self.idx < len(self.replies):
                replies = [self.replies[self.idx]]
                self.idx += 1
            else:
                self.idx = 0
                replies = [self.replies[self.idx]]
                self.idx += 1

            return {"replies": replies}

    @component
    class AnswerBuilderWithPrompt:
        @component.output_types(answers=List[GeneratedAnswer])
        def run(self, replies: List[str], query: str, prompt: Optional[str] = None) -> Dict[str, Any]:
            answer = GeneratedAnswer(data=replies[0], query=query, documents=[])

            if prompt is not None:
                answer.meta["prompt"] = prompt

            return {"answers": [answer]}

    code_prompt_template = "{{task}}"

    feedback_prompt_template = """
Check if this code is valid and can run: {{ code[0] }}
Return "PASS" if it passes and "FAIL" if it fails.
Provide additional feedback on why it fails.
    """

    valid_response = """
def generate_santa_sleigh():
    '''
    Returns ASCII art of Santa Claus on his sleigh with Rudolph leading the way.
    '''
    # implementation goes here.
    return art
    """

    code_llm = FixedGenerator(replies=["invalid code", "invalid code", valid_response])
    code_prompt = PromptBuilder(template=code_prompt_template)

    feedback_llm = FixedGenerator(replies=["FAIL", "FAIL, come on, try again.", "PASS"])
    feedback_prompt = PromptBuilder(template=feedback_prompt_template)

    routes = [
        {
            "condition": "{{ 'FAIL' in replies[0] }}",
            "output": "{{ replies[0] }}",
            "output_name": "fail",
            "output_type": str,
        },
        {
            "condition": "{{ 'PASS' in replies[0] }}",
            "output": "{{ code }}",
            "output_name": "pass",
            "output_type": List[str],
        },
    ]

    router = ConditionalRouter(routes=routes)
    joiner = BranchJoiner(type_=str)
    concatenator = OutputAdapter(
        template="{{code_prompt + '\n' + generated_code[0] + '\n' + feedback}}", output_type=str
    )

    answer_builder = AnswerBuilderWithPrompt()

    pp = pipeline_class(max_runs_per_component=100)

    pp.add_component("concatenator", concatenator)
    pp.add_component("code_llm", code_llm)
    pp.add_component("code_prompt", code_prompt)
    pp.add_component("feedback_prompt", feedback_prompt)
    pp.add_component("feedback_llm", feedback_llm)
    pp.add_component("router", router)
    pp.add_component("joiner", joiner)

    pp.add_component("answer_builder", answer_builder)

    pp.connect("concatenator.output", "joiner.value")
    pp.connect("joiner.value", "code_prompt.task")
    pp.connect("code_prompt.prompt", "code_llm.prompt")
    pp.connect("code_prompt.prompt", "concatenator.code_prompt")
    pp.connect("code_llm.replies", "feedback_prompt.code")
    pp.connect("feedback_llm.replies", "router.replies")
    pp.connect("router.fail", "concatenator.feedback")
    pp.connect("feedback_prompt.prompt", "feedback_llm.prompt")
    pp.connect("router.pass", "answer_builder.replies")
    pp.connect("code_llm.replies", "router.code")
    pp.connect("code_llm.replies", "concatenator.generated_code")
    pp.connect("concatenator.output", "answer_builder.prompt")

    task = "Generate code to generate christmas ascii-art"

    expected_prompt = """Generate code to generate christmas ascii-art
invalid code
FAIL
invalid code
FAIL, come on, try again."""
    return (
        pp,
        [
            PipelineRunData(
                inputs={"joiner": {"value": task}, "answer_builder": {"query": task}},
                expected_outputs={
                    "answer_builder": {
                        "answers": [
                            GeneratedAnswer(
                                data=valid_response, query=task, documents=[], meta={"prompt": expected_prompt}
                            )
                        ]
                    }
                },
                expected_component_calls={
                    ("answer_builder", 1): {
                        "prompt": "Generate code to generate christmas "
                        "ascii-art\n"
                        "invalid code\n"
                        "FAIL\n"
                        "invalid code\n"
                        "FAIL, come on, try again.",
                        "query": "Generate code to generate christmas ascii-art",
                        "replies": [
                            "\n"
                            "def generate_santa_sleigh():\n"
                            "    '''\n"
                            "    Returns ASCII art of Santa Claus on "
                            "his sleigh with Rudolph leading the "
                            "way.\n"
                            "    '''\n"
                            "    # implementation goes here.\n"
                            "    return art\n"
                            "    "
                        ],
                    },
                    ("code_llm", 1): {"prompt": "Generate code to generate christmas ascii-art"},
                    ("code_llm", 2): {"prompt": "Generate code to generate christmas ascii-art\ninvalid code\nFAIL"},
                    ("code_llm", 3): {
                        "prompt": "Generate code to generate christmas ascii-art\n"
                        "invalid code\n"
                        "FAIL\n"
                        "invalid code\n"
                        "FAIL, come on, try again."
                    },
                    ("code_prompt", 1): {
                        "task": "Generate code to generate christmas ascii-art",
                        "template": None,
                        "template_variables": None,
                    },
                    ("code_prompt", 2): {
                        "task": "Generate code to generate christmas ascii-art\ninvalid code\nFAIL",
                        "template": None,
                        "template_variables": None,
                    },
                    ("code_prompt", 3): {
                        "task": "Generate code to generate christmas ascii-art\n"
                        "invalid code\n"
                        "FAIL\n"
                        "invalid code\n"
                        "FAIL, come on, try again.",
                        "template": None,
                        "template_variables": None,
                    },
                    ("concatenator", 1): {
                        "code_prompt": "Generate code to generate christmas ascii-art",
                        "feedback": "FAIL",
                        "generated_code": ["invalid code"],
                    },
                    ("concatenator", 2): {
                        "code_prompt": "Generate code to generate christmas ascii-art\ninvalid code\nFAIL",
                        "feedback": "FAIL, come on, try again.",
                        "generated_code": ["invalid code"],
                    },
                    ("feedback_llm", 1): {
                        "prompt": "\n"
                        "Check if this code is valid and can run: "
                        "invalid code\n"
                        'Return "PASS" if it passes and "FAIL" if it '
                        "fails.\n"
                        "Provide additional feedback on why it "
                        "fails.\n"
                        "    "
                    },
                    ("feedback_llm", 2): {
                        "prompt": "\n"
                        "Check if this code is valid and can run: "
                        "invalid code\n"
                        'Return "PASS" if it passes and "FAIL" if it '
                        "fails.\n"
                        "Provide additional feedback on why it "
                        "fails.\n"
                        "    "
                    },
                    ("feedback_llm", 3): {
                        "prompt": "\n"
                        "Check if this code is valid and can run: \n"
                        "def generate_santa_sleigh():\n"
                        "    '''\n"
                        "    Returns ASCII art of Santa Claus on his "
                        "sleigh with Rudolph leading the way.\n"
                        "    '''\n"
                        "    # implementation goes here.\n"
                        "    return art\n"
                        "    \n"
                        'Return "PASS" if it passes and "FAIL" if it '
                        "fails.\n"
                        "Provide additional feedback on why it "
                        "fails.\n"
                        "    "
                    },
                    ("feedback_prompt", 1): {"code": ["invalid code"], "template": None, "template_variables": None},
                    ("feedback_prompt", 2): {"code": ["invalid code"], "template": None, "template_variables": None},
                    ("feedback_prompt", 3): {
                        "code": [
                            "\n"
                            "def generate_santa_sleigh():\n"
                            "    '''\n"
                            "    Returns ASCII art of Santa Claus on "
                            "his sleigh with Rudolph leading the way.\n"
                            "    '''\n"
                            "    # implementation goes here.\n"
                            "    return art\n"
                            "    "
                        ],
                        "template": None,
                        "template_variables": None,
                    },
                    ("joiner", 1): {"value": ["Generate code to generate christmas ascii-art"]},
                    ("joiner", 2): {"value": ["Generate code to generate christmas ascii-art\ninvalid code\nFAIL"]},
                    ("joiner", 3): {
                        "value": [
                            "Generate code to generate christmas ascii-art\n"
                            "invalid code\n"
                            "FAIL\n"
                            "invalid code\n"
                            "FAIL, come on, try again."
                        ]
                    },
                    ("router", 1): {"code": ["invalid code"], "replies": ["FAIL"]},
                    ("router", 2): {"code": ["invalid code"], "replies": ["FAIL, come on, try again."]},
                    ("router", 3): {
                        "code": [
                            "\n"
                            "def generate_santa_sleigh():\n"
                            "    '''\n"
                            "    Returns ASCII art of Santa Claus on his sleigh "
                            "with Rudolph leading the way.\n"
                            "    '''\n"
                            "    # implementation goes here.\n"
                            "    return art\n"
                            "    "
                        ],
                        "replies": ["PASS"],
                    },
                },
            )
        ],
    )


@given("a pipeline with a component that has dynamic default inputs", target_fixture="pipeline_data")
def pipeline_with_dynamic_defaults(pipeline_class):
    @component
    class ParrotWithDynamicDefaultInputs:
        def __init__(self, input_variable: str):
            self.input_variable = input_variable
            component.set_input_type(self, input_variable, str, default="Parrot doesn't only parrot!")

        @component.output_types(response=str)
        def run(self, **kwargs):
            return {"response": kwargs[self.input_variable]}

    parrot = ParrotWithDynamicDefaultInputs("parrot")
    pipeline = pipeline_class()
    pipeline.add_component("parrot", parrot)
    return (
        pipeline,
        [
            PipelineRunData(
                inputs={"parrot": {"parrot": "Are you a parrot?"}},
                expected_outputs={"parrot": {"response": "Are you a parrot?"}},
                expected_component_calls={("parrot", 1): {"parrot": "Are you a parrot?"}},
            ),
            PipelineRunData(
                inputs={},
                expected_outputs={"parrot": {"response": "Parrot doesn't only parrot!"}},
                expected_component_calls={("parrot", 1): {"parrot": "Parrot doesn't only parrot!"}},
            ),
        ],
    )


@given("a pipeline with a component that has variadic dynamic default inputs", target_fixture="pipeline_data")
def pipeline_with_variadic_dynamic_defaults(pipeline_class):
    @component
    class ParrotWithVariadicDynamicDefaultInputs:
        def __init__(self, input_variable: str):
            self.input_variable = input_variable
            component.set_input_type(self, input_variable, Variadic[str], default="Parrot doesn't only parrot!")

        @component.output_types(response=List[str])
        def run(self, **kwargs):
            return {"response": kwargs[self.input_variable]}

    parrot = ParrotWithVariadicDynamicDefaultInputs("parrot")
    pipeline = pipeline_class()
    pipeline.add_component("parrot", parrot)
    return (
        pipeline,
        [
            PipelineRunData(
                inputs={"parrot": {"parrot": "Are you a parrot?"}},
                expected_outputs={"parrot": {"response": ["Are you a parrot?"]}},
                expected_component_calls={("parrot", 1): {"parrot": ["Are you a parrot?"]}},
            ),
            PipelineRunData(
                inputs={},
                expected_outputs={"parrot": {"response": ["Parrot doesn't only parrot!"]}},
                expected_component_calls={("parrot", 1): {"parrot": ["Parrot doesn't only parrot!"]}},
            ),
        ],
    )


@given("a pipeline that is a file conversion pipeline with two joiners", target_fixture="pipeline_data")
def pipeline_that_converts_files(pipeline_class):
    csv_data = """
some,header,row
0,1,0
    """

    txt_data = "Text file content for testing this."

    json_data = '{"content": "Some test content"}'

    sources = [
        ByteStream.from_string(text=csv_data, mime_type="text/csv", meta={"file_type": "csv"}),
        ByteStream.from_string(text=txt_data, mime_type="text/plain", meta={"file_type": "txt"}),
        ByteStream.from_string(text=json_data, mime_type="application/json", meta={"file_type": "json"}),
    ]

    router = FileTypeRouter(mime_types=["text/csv", "text/plain", "application/json"])
    splitter = DocumentSplitter(split_by="word", split_length=3, split_overlap=0)
    txt_converter = TextFileToDocument()
    csv_converter = CSVToDocument()
    json_converter = JSONConverter(content_key="content")

    b_joiner = DocumentJoiner()
    a_joiner = DocumentJoiner()

    pp = pipeline_class(max_runs_per_component=1)

    pp.add_component("router", router)
    pp.add_component("splitter", splitter)
    pp.add_component("txt_converter", txt_converter)
    pp.add_component("csv_converter", csv_converter)
    pp.add_component("json_converter", json_converter)
    pp.add_component("b_joiner", b_joiner)
    pp.add_component("a_joiner", a_joiner)

    pp.connect("router.text/plain", "txt_converter.sources")
    pp.connect("router.application/json", "json_converter.sources")
    pp.connect("router.text/csv", "csv_converter.sources")
    pp.connect("txt_converter.documents", "b_joiner.documents")
    pp.connect("json_converter.documents", "b_joiner.documents")
    pp.connect("csv_converter.documents", "a_joiner.documents")
    pp.connect("b_joiner.documents", "splitter.documents")
    pp.connect("splitter.documents", "a_joiner.documents")

    expected_pre_split_docs = [
        Document(content="Some test content", meta={"file_type": "json"}),
        Document(content=txt_data, meta={"file_type": "txt"}),
    ]
    expected_splits_docs = [
        Document(
            content="Some test content",
            meta={
                "file_type": "json",
                "source_id": "0c6c5951d18da2935c7af3e24d417a9f94ca85403866dcfee1de93922504e1e5",
                "page_number": 1,
                "split_id": 0,
                "split_idx_start": 0,
            },
        ),
        Document(
            content="Text file content ",
            meta={
                "file_type": "txt",
                "source_id": "41cb91740f6e64ab542122936ea746c238ae0a92fd29b698efabbe23d0ba4c42",
                "page_number": 1,
                "split_id": 0,
                "split_idx_start": 0,
            },
        ),
        Document(
            content="for testing this.",
            meta={
                "file_type": "txt",
                "source_id": "41cb91740f6e64ab542122936ea746c238ae0a92fd29b698efabbe23d0ba4c42",
                "page_number": 1,
                "split_id": 1,
                "split_idx_start": 18,
            },
        ),
    ]

    expected_csv_docs = [Document(content=csv_data, meta={"file_type": "csv"})]

    return (
        pp,
        [
            PipelineRunData(
                inputs={"router": {"sources": sources}},
                expected_outputs={"a_joiner": {"documents": expected_csv_docs + expected_splits_docs}},
                expected_component_calls={
                    ("router", 1): {"sources": sources, "meta": None},
                    ("csv_converter", 1): {"sources": [sources[0]], "meta": None},
                    ("txt_converter", 1): {"sources": [sources[1]], "meta": None},
                    ("json_converter", 1): {"sources": [sources[2]], "meta": None},
                    ("b_joiner", 1): {
                        "documents": [[expected_pre_split_docs[0]], [expected_pre_split_docs[1]]],
                        "top_k": None,
                    },
                    ("splitter", 1): {"documents": expected_pre_split_docs},
                    ("a_joiner", 1): {"documents": [expected_csv_docs, expected_splits_docs], "top_k": None},
                },
            )
        ],
    )


@given("a pipeline that is a file conversion pipeline with three joiners", target_fixture="pipeline_data")
def pipeline_that_converts_files_with_three_joiners(pipeline_class):
    # What does this test?
    # When a component does not produce outputs, and the successors only receive inputs from this component,
    # then the successors will not run.
    # The successor of the successor would never know that its predecessor did not run if we don't send a signal.
    # This is why we use PipelineBase._notify_downstream_components to recursively notify successors that
    # can not run anymore.
    # This prevents an edge case where multiple lazy variadic components wait for input and the execution order
    # would otherwise be decided by lexicographical sort.
    html_data = """
<html><body>Some content</body></html>
    """

    txt_data = "Text file content"

    sources = [
        ByteStream.from_string(text=txt_data, mime_type="text/plain", meta={"file_type": "txt"}),
        ByteStream.from_string(text=html_data, mime_type="text/html", meta={"file_type": "html"}),
    ]

    router = FileTypeRouter(mime_types=["text/csv", "text/plain", "application/json", "text/html"])
    splitter = DocumentSplitter(split_by="word", split_length=3, split_overlap=0)
    page_splitter = DocumentSplitter(split_by="page", split_length=1, split_overlap=0)
    txt_converter = TextFileToDocument()
    csv_converter = CSVToDocument()
    json_converter = JSONConverter(content_key="content")
    html_converter = HTMLToDocument()

    DocumentJoiner_1 = DocumentJoiner()
    joiner = DocumentJoiner()
    DocumentJoiner_2 = DocumentJoiner()

    pp = pipeline_class(max_runs_per_component=2)

    pp.add_component("router", router)
    pp.add_component("splitter", splitter)
    pp.add_component("txt_converter", txt_converter)
    pp.add_component("csv_converter", csv_converter)
    pp.add_component("json_converter", json_converter)
    pp.add_component("html_converter", html_converter)
    pp.add_component("joiner", joiner)
    pp.add_component("DocumentJoiner_1", DocumentJoiner_1)
    pp.add_component("DocumentJoiner_2", DocumentJoiner_2)
    pp.add_component("page_splitter", page_splitter)

    pp.connect("router.text/plain", "txt_converter.sources")
    pp.connect("router.application/json", "json_converter.sources")
    pp.connect("router.text/csv", "csv_converter.sources")
    pp.connect("router.text/html", "html_converter.sources")
    pp.connect("txt_converter.documents", "joiner.documents")
    pp.connect("json_converter.documents", "joiner.documents")
    pp.connect("csv_converter.documents", "DocumentJoiner_1.documents")
    pp.connect("html_converter.documents", "DocumentJoiner_1.documents")
    pp.connect("joiner.documents", "splitter.documents")
    pp.connect("DocumentJoiner_1.documents", "page_splitter.documents")
    pp.connect("splitter.documents", "DocumentJoiner_2.documents")
    pp.connect("page_splitter.documents", "DocumentJoiner_2.documents")

    expected_html_doc = Document(content="Some content", meta={"file_type": "html"})
    expected_txt_doc = Document(content=txt_data, meta={"file_type": "txt"})
    expected_txt_split_doc = Document(
        content=txt_data,
        meta={
            "file_type": "txt",
            "source_id": expected_txt_doc.id,
            "page_number": 1,
            "split_id": 0,
            "split_idx_start": 0,
        },
    )
    expected_html_split_doc = Document(
        content=expected_html_doc.content,
        meta={
            "file_type": "html",
            "source_id": expected_html_doc.id,
            "page_number": 1,
            "split_id": 0,
            "split_idx_start": 0,
        },
    )

    return (
        pp,
        [
            PipelineRunData(
                inputs={"router": {"sources": sources}},
                # HTML converter takes longer than TXT Converter and this is why the order of documents is not stable
                # for AsyncPipeline. We test for ANY here to make the test pass.
                # In real usage, if the user cares about the order of documents arriving at a lazy variadic component,
                # the user should pick a different component (OutputAdapter) to combine the lists.
                expected_outputs={"DocumentJoiner_2": {"documents": ANY}},
                expected_component_calls={
                    ("router", 1): {"sources": sources, "meta": None},
                    ("html_converter", 1): {"sources": [sources[1]], "meta": None, "extraction_kwargs": None},
                    ("txt_converter", 1): {"sources": [sources[0]], "meta": None},
                    ("joiner", 1): {"documents": [[expected_txt_doc]], "top_k": None},
                    ("DocumentJoiner_1", 1): {"documents": [[expected_html_doc]], "top_k": None},
                    # Same as above
                    ("DocumentJoiner_2", 1): {"documents": ANY, "top_k": None},
                    ("splitter", 1): {"documents": [expected_txt_doc]},
                    ("page_splitter", 1): {"documents": [expected_html_doc]},
                },
            )
        ],
    )


@given("a pipeline that is a file conversion pipeline with three joiners and a loop", target_fixture="pipeline_data")
def pipeline_that_converts_files_with_three_joiners_and_a_loop(pipeline_class):
    # What does this test?
    # When a component does not produce outputs, and the successors only receive inputs from this component,
    # then the successors will not run.
    # The successor of the successor would never know that its predecessor did not run if we don't send a signal.
    # This is why we use PipelineBase._notify_downstream_components to recursively notify successors that
    # can not run anymore.
    # This prevents an edge case where multiple lazy variadic components wait for input and the execution order
    # would otherwise be decided by lexicographical sort.
    @component
    class FakeDataExtractor:
        def __init__(self, metas):
            self.metas = metas
            self.current_idx = 0

        @component.output_types(documents=List[Document])
        def run(self, documents: List[Document]):
            sorted_docs = sorted(documents, key=lambda doc: doc.meta["file_type"])
            if self.current_idx >= len(sorted_docs):
                self.current_idx = 0

            for doc in sorted_docs:
                doc.meta = {**doc.meta, **self.metas[self.current_idx]}
            self.current_idx += 1

            return {"documents": sorted_docs}

    html_data = """
<html><body>Some content</body></html>
    """

    txt_data = "Text file content"

    sources = [
        ByteStream.from_string(text=txt_data, mime_type="text/plain", meta={"file_type": "txt"}),
        ByteStream.from_string(text=html_data, mime_type="text/html", meta={"file_type": "html"}),
    ]

    router = FileTypeRouter(mime_types=["text/csv", "text/plain", "application/json", "text/html"])
    splitter = DocumentSplitter(split_by="word", split_length=3, split_overlap=0)
    page_splitter = DocumentSplitter(split_by="page", split_length=1, split_overlap=0)
    txt_converter = TextFileToDocument()
    csv_converter = CSVToDocument()
    json_converter = JSONConverter(content_key="content")
    html_converter = HTMLToDocument()

    DocumentJoiner_1 = DocumentJoiner()
    joiner = DocumentJoiner()
    DocumentJoiner_2 = DocumentJoiner()

    extraction_routes = [
        {
            "condition": "{{documents[0].meta['iteration'] == 1}}",
            "output_name": "continue",
            "output": "{{documents}}",
            "output_type": List[Document],
        },
        {
            "condition": "{{documents[0].meta['iteration'] == 2}}",
            "output_name": "stop",
            "output": "{{documents}}",
            "output_type": List[Document],
        },
    ]

    extraction_router = ConditionalRouter(routes=extraction_routes, unsafe=True)

    pp = pipeline_class(max_runs_per_component=4)

    pp.add_component("router", router)
    pp.add_component("splitter", splitter)
    pp.add_component("txt_converter", txt_converter)
    pp.add_component("csv_converter", csv_converter)
    pp.add_component("json_converter", json_converter)
    pp.add_component("html_converter", html_converter)
    pp.add_component("joiner", joiner)
    pp.add_component("DocumentJoiner_1", DocumentJoiner_1)
    pp.add_component("DocumentJoiner_2", DocumentJoiner_2)
    pp.add_component("page_splitter", page_splitter)
    pp.add_component("metadata_generator", FakeDataExtractor(metas=[{"iteration": 1}, {"iteration": 2}]))
    pp.add_component("extraction_router", extraction_router)
    pp.add_component("branch_joiner", BranchJoiner(type_=List[Document]))

    pp.connect("router.text/plain", "txt_converter.sources")
    pp.connect("router.application/json", "json_converter.sources")
    pp.connect("router.text/csv", "csv_converter.sources")
    pp.connect("router.text/html", "html_converter.sources")
    pp.connect("txt_converter.documents", "joiner.documents")
    pp.connect("json_converter.documents", "joiner.documents")
    pp.connect("csv_converter.documents", "DocumentJoiner_1.documents")
    pp.connect("html_converter.documents", "DocumentJoiner_1.documents")
    pp.connect("joiner.documents", "splitter.documents")
    pp.connect("DocumentJoiner_1.documents", "page_splitter.documents")
    pp.connect("splitter.documents", "DocumentJoiner_2.documents")
    pp.connect("page_splitter.documents", "DocumentJoiner_2.documents")
    pp.connect("DocumentJoiner_2.documents", "branch_joiner.value")
    pp.connect("branch_joiner.value", "metadata_generator.documents")
    pp.connect("metadata_generator.documents", "extraction_router.documents")
    pp.connect("extraction_router.continue", "branch_joiner.value")

    expected_html_doc = Document(content="Some content", meta={"file_type": "html"})
    expected_txt_doc = Document(content=txt_data, meta={"file_type": "txt"})
    expected_docs = [
        Document(
            content=expected_html_doc.content,
            meta={
                "file_type": "html",
                "source_id": expected_html_doc.id,
                "page_number": 1,
                "split_id": 0,
                "split_idx_start": 0,
            },
        ),
        Document(
            content=txt_data,
            meta={
                "file_type": "txt",
                "source_id": expected_txt_doc.id,
                "page_number": 1,
                "split_id": 0,
                "split_idx_start": 0,
            },
        ),
    ]

    expected_docs_iteration_1 = []
    expected_docs_iteration_2 = []
    for doc in expected_docs:
        doc_1 = deepcopy(doc)
        doc_1.meta["iteration"] = 1
        doc_2 = deepcopy(doc)
        doc_2.meta["iteration"] = 2
        expected_docs_iteration_1.append(doc_1)
        expected_docs_iteration_2.append(doc_2)

    return (
        pp,
        [
            PipelineRunData(
                inputs={"router": {"sources": sources}},
                expected_outputs={"extraction_router": {"stop": expected_docs_iteration_2}},
                expected_component_calls={
                    ("router", 1): {"sources": sources, "meta": None},
                    ("html_converter", 1): {"sources": [sources[1]], "meta": None, "extraction_kwargs": None},
                    ("txt_converter", 1): {"sources": [sources[0]], "meta": None},
                    ("joiner", 1): {"documents": [[expected_txt_doc]], "top_k": None},
                    ("DocumentJoiner_1", 1): {"documents": [[expected_html_doc]], "top_k": None},
                    # ANY because we can't know the order of documents for AsyncPipeline
                    ("DocumentJoiner_2", 1): {"documents": ANY, "top_k": None},
                    ("splitter", 1): {"documents": [expected_txt_doc]},
                    ("page_splitter", 1): {"documents": [expected_html_doc]},
                    # Same as above
                    ("branch_joiner", 1): {"value": ANY},
                    # Same as above
                    ("metadata_generator", 1): {"documents": ANY},
                    ("extraction_router", 1): {"documents": expected_docs_iteration_1},
                    ("branch_joiner", 2): {"value": [expected_docs_iteration_1]},
                    ("metadata_generator", 2): {"documents": expected_docs_iteration_1},
                    ("extraction_router", 2): {"documents": expected_docs_iteration_2},
                },
            )
        ],
    )


@given("a pipeline that has components returning dataframes", target_fixture="pipeline_data")
def pipeline_has_components_returning_dataframes(pipeline_class):
    def get_df():
        return pd.DataFrame({"a": [1, 2], "b": [1, 2]})

    @component
    class DataFramer:
        @component.output_types(dataframe=pd.DataFrame)
        def run(self, dataframe: pd.DataFrame) -> Dict[str, Any]:
            return {"dataframe": get_df()}

    pp = pipeline_class(max_runs_per_component=1)

    pp.add_component("df_1", DataFramer())
    pp.add_component("df_2", DataFramer())

    pp.connect("df_1", "df_2")

    return (
        pp,
        [
            PipelineRunData(
                inputs={"df_1": {"dataframe": get_df()}},
                expected_outputs={"df_2": {"dataframe": get_df()}},
                expected_component_calls={("df_1", 1): {"dataframe": get_df()}, ("df_2", 1): {"dataframe": get_df()}},
            )
        ],
    )
