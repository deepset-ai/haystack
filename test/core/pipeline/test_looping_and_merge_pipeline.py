# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import logging

from haystack.components.others import Multiplexer
from haystack.core.pipeline import Pipeline
from haystack.testing.sample_components import Accumulate, AddFixedValue, Sum, Threshold

logging.basicConfig(level=logging.DEBUG)


def test_pipeline_fixed():
    accumulator = Accumulate()
    pipeline = Pipeline(max_loops_allowed=10)
    add_zero = AddFixedValue(add=0)
    multiplexer = Multiplexer(type_=int)
    sum = Sum()
    below_10 = Threshold(threshold=10)
    add_one = AddFixedValue(add=1)
    counter = accumulator
    add_two = AddFixedValue(add=2)
    pipeline.add_component("add_zero", add_zero)
    pipeline.add_component("multiplexer", multiplexer)
    pipeline.add_component("sum", sum)
    pipeline.add_component("below_10", below_10)
    pipeline.add_component("add_one", add_one)
    pipeline.add_component("counter", counter)
    pipeline.add_component("add_two", add_two)

    pipeline.connect(add_zero.outputs.result, multiplexer.inputs.value)
    pipeline.connect(multiplexer.outputs.value, below_10.inputs.value)
    pipeline.connect(below_10.outputs.below, add_one.inputs.value)
    pipeline.connect(add_one.outputs.result, counter.inputs.value)
    pipeline.connect(counter.outputs.value, multiplexer.inputs.value)
    pipeline.connect(below_10.outputs.above, add_two.inputs.value)
    pipeline.connect(add_two.outputs.result, sum.inputs.values)

    results = pipeline.run({"add_zero": {"value": 8}, "sum": {"values": 2}})
    assert results == {"sum": {"total": 23}}
    assert accumulator.state == 19


def test_pipeline_variadic():
    accumulator = Accumulate()
    pipeline = Pipeline(max_loops_allowed=10)
    add_zero = AddFixedValue(add=0)
    multiplexer = Multiplexer(type_=int)
    sum = Sum()
    below_10 = Threshold(threshold=10)
    add_one = AddFixedValue(add=1)
    counter = accumulator
    add_two = AddFixedValue(add=2)
    pipeline.add_component("add_zero", add_zero)
    pipeline.add_component("multiplexer", multiplexer)
    pipeline.add_component("sum", sum)
    pipeline.add_component("below_10", below_10)
    pipeline.add_component("add_one", add_one)
    pipeline.add_component("counter", counter)
    pipeline.add_component("add_two", add_two)

    pipeline.connect(add_zero.outputs.result, multiplexer.inputs.value)
    pipeline.connect(multiplexer.outputs.value, below_10.inputs.value)
    pipeline.connect(below_10.outputs.below, add_one.inputs.value)
    pipeline.connect(add_one.outputs.result, counter.inputs.value)
    pipeline.connect(counter.outputs.value, multiplexer.inputs.value)
    pipeline.connect(below_10.outputs.above, add_two.inputs.value)
    pipeline.connect(add_two.outputs.result, sum.inputs.values)

    results = pipeline.run({"add_zero": {"value": 8}, "sum": {"values": 2}})
    assert results == {"sum": {"total": 23}}
    assert accumulator.state == 19
