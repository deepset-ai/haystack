# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import logging

from haystack.components.others import Multiplexer
from haystack.core.pipeline import Pipeline
from haystack.testing.sample_components import Accumulate, AddFixedValue, Threshold

logging.basicConfig(level=logging.DEBUG)


def test_pipeline():
    pipeline = Pipeline(max_loops_allowed=10)
    add_one = AddFixedValue(add=1)
    multiplexer = Multiplexer(type_=int)
    below_10 = Threshold(threshold=10)
    accumulator = Accumulate()
    add_two = AddFixedValue(add=2)
    pipeline.add_component("add_one", add_one)
    pipeline.add_component("multiplexer", multiplexer)
    pipeline.add_component("below_10", below_10)
    pipeline.add_component("accumulator", accumulator)
    pipeline.add_component("add_two", add_two)

    pipeline.connect(add_one.outputs.result, multiplexer.inputs.value)
    pipeline.connect(multiplexer.outputs.value, below_10.inputs.value)
    pipeline.connect(below_10.outputs.below, accumulator.inputs.value)
    pipeline.connect(accumulator.outputs.value, multiplexer.inputs.value)
    pipeline.connect(below_10.outputs.above, add_two.inputs.value)

    results = pipeline.run({"add_one": {"value": 3}})
    assert results == {"add_two": {"result": 18}}
    assert accumulator.state == 16


def test_pipeline_direct_io_loop():
    pipeline = Pipeline(max_loops_allowed=10)
    multiplexer = Multiplexer(type_=int)
    below_10 = Threshold(threshold=10)
    accumulator = Accumulate()
    pipeline.add_component("multiplexer", multiplexer)
    pipeline.add_component("below_10", below_10)
    pipeline.add_component("accumulator", accumulator)

    pipeline.connect(multiplexer.outputs.value, below_10.inputs.value)
    pipeline.connect(below_10.outputs.below, accumulator.inputs.value)
    pipeline.connect(accumulator.outputs.value, multiplexer.inputs.value)

    results = pipeline.run({"multiplexer": {"value": 4}})
    assert results == {"below_10": {"above": 16}}
    assert accumulator.state == 16


def test_pipeline_fixed_merger_input():
    pipeline = Pipeline(max_loops_allowed=10)
    multiplexer = Multiplexer(type_=int)
    below_10 = Threshold(threshold=10)
    accumulator = Accumulate()
    add_two = AddFixedValue(add=2)
    pipeline.add_component("multiplexer", multiplexer)
    pipeline.add_component("below_10", below_10)
    pipeline.add_component("accumulator", accumulator)
    pipeline.add_component("add_two", add_two)

    pipeline.connect(multiplexer.outputs.value, below_10.inputs.value)
    pipeline.connect(below_10.outputs.below, accumulator.inputs.value)
    pipeline.connect(accumulator.outputs.value, multiplexer.inputs.value)
    pipeline.connect(below_10.outputs.above, add_two.inputs.value)

    results = pipeline.run({"multiplexer": {"value": 4}})
    assert results == {"add_two": {"result": 18}}
    assert accumulator.state == 16


def test_pipeline_variadic_merger_input():
    pipeline = Pipeline(max_loops_allowed=10)
    multiplexer = Multiplexer(type_=int)
    below_10 = Threshold(threshold=10)
    accumulator = Accumulate()
    add_two = AddFixedValue(add=2)
    pipeline.add_component("multiplexer", multiplexer)
    pipeline.add_component("below_10", below_10)
    pipeline.add_component("accumulator", accumulator)
    pipeline.add_component("add_two", add_two)

    pipeline.connect(multiplexer.outputs.value, below_10.inputs.value)
    pipeline.connect(below_10.outputs.below, accumulator.inputs.value)
    pipeline.connect(accumulator.outputs.value, multiplexer.inputs.value)
    pipeline.connect(below_10.outputs.above, add_two.inputs.value)

    results = pipeline.run({"multiplexer": {"value": 4}})
    assert results == {"add_two": {"result": 18}}
    assert accumulator.state == 16
