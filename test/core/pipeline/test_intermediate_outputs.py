# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import logging

from haystack.components.others import Multiplexer
from haystack.core.pipeline import Pipeline
from haystack.core.component import component
from haystack.testing.sample_components import Accumulate, AddFixedValue, Double, Threshold

logging.basicConfig(level=logging.DEBUG)


@component
class DoubleWithOriginal:
    """
    Doubles the input value and returns the original value as well.
    """

    @component.output_types(value=int, original=int)
    def run(self, value: int):
        return {"value": value * 2, "original": value}


def test_pipeline_intermediate_outputs():
    pipeline = Pipeline()
    pipeline.add_component("first_addition", AddFixedValue(add=2))
    pipeline.add_component("second_addition", AddFixedValue())
    pipeline.add_component("double", Double())
    pipeline.connect("first_addition", "double")
    pipeline.connect("double", "second_addition")

    results = pipeline.run(
        {"first_addition": {"value": 1}}, include_outputs_from={"first_addition", "second_addition", "double"}
    )
    assert results == {"second_addition": {"result": 7}, "first_addition": {"result": 3}, "double": {"value": 6}}

    results = pipeline.run({"first_addition": {"value": 1}}, include_outputs_from={"double"})
    assert results == {"second_addition": {"result": 7}, "double": {"value": 6}}


def test_pipeline_with_loops_intermediate_outputs():
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

    results = pipeline.run(
        {"add_one": {"value": 3}},
        include_outputs_from={"add_two", "add_one", "multiplexer", "below_10", "accumulator", "below_5", "add_three"},
    )

    assert results == {
        "add_two": {"result": 13},
        "add_one": {"result": 4},
        "multiplexer": {"value": 11},
        "below_10": {"above": 11},
        "accumulator": {"value": 8},
        "below_5": {"above": 8},
        "add_three": {"result": 11},
    }


def test_pipeline_intermediate_outputs_multiple_output_sockets():
    pipeline = Pipeline()
    pipeline.add_component("first_addition", AddFixedValue(add=2))
    pipeline.add_component("second_addition", AddFixedValue())
    pipeline.add_component("double", DoubleWithOriginal())
    pipeline.connect("first_addition", "double")
    pipeline.connect("double.value", "second_addition")

    results = pipeline.run(
        {"first_addition": {"value": 1}}, include_outputs_from={"first_addition", "second_addition", "double"}
    )
    assert results == {
        "second_addition": {"result": 7},
        "first_addition": {"result": 3},
        "double": {"value": 6, "original": 3},
    }

    results = pipeline.run({"first_addition": {"value": 1}}, include_outputs_from={"double"})
    assert results == {"second_addition": {"result": 7}, "double": {"value": 6, "original": 3}}
