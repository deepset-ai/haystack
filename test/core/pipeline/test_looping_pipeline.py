# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import logging

from haystack.components.others import Multiplexer
from haystack.core.pipeline import Pipeline
from haystack.testing.sample_components import Accumulate, AddFixedValue, FirstIntSelector, Threshold

logging.basicConfig(level=logging.DEBUG)


def test_pipeline():
    accumulator = Accumulate()

    pipeline = Pipeline(max_loops_allowed=10)
    pipeline.add_component("add_one", AddFixedValue(add=1))
    pipeline.add_component("multiplexer", Multiplexer(type_=int))
    pipeline.add_component("below_10", Threshold(threshold=10))
    pipeline.add_component("accumulator", accumulator)
    pipeline.add_component("add_two", AddFixedValue(add=2))

    pipeline.connect("add_one.result", "multiplexer")
    pipeline.connect("multiplexer.value", "below_10.value")
    pipeline.connect("below_10.below", "accumulator.value")
    pipeline.connect("accumulator.value", "multiplexer")
    pipeline.connect("below_10.above", "add_two.value")

    results = pipeline.run({"add_one": {"value": 3}})
    assert results == {"add_two": {"result": 18}}
    assert accumulator.state == 16


def test_pipeline_direct_io_loop():
    accumulator = Accumulate()

    pipeline = Pipeline(max_loops_allowed=10)
    pipeline.add_component("multiplexer", Multiplexer(type_=int))
    pipeline.add_component("below_10", Threshold(threshold=10))
    pipeline.add_component("accumulator", accumulator)

    pipeline.connect("multiplexer.value", "below_10.value")
    pipeline.connect("below_10.below", "accumulator.value")
    pipeline.connect("accumulator.value", "multiplexer")

    results = pipeline.run({"multiplexer": {"value": 4}})
    assert results == {"below_10": {"above": 16}}
    assert accumulator.state == 16


def test_pipeline_fixed_merger_input():
    accumulator = Accumulate()

    pipeline = Pipeline(max_loops_allowed=10)
    pipeline.add_component("multiplexer", Multiplexer(type_=int))
    pipeline.add_component("below_10", Threshold(threshold=10))
    pipeline.add_component("accumulator", accumulator)
    pipeline.add_component("add_two", AddFixedValue(add=2))

    pipeline.connect("multiplexer.value", "below_10.value")
    pipeline.connect("below_10.below", "accumulator.value")
    pipeline.connect("accumulator.value", "multiplexer")
    pipeline.connect("below_10.above", "add_two.value")

    results = pipeline.run({"multiplexer": {"value": 4}})
    assert results == {"add_two": {"result": 18}}
    assert accumulator.state == 16


def test_pipeline_variadic_merger_input():
    accumulator = Accumulate()

    pipeline = Pipeline(max_loops_allowed=10)
    pipeline.add_component("merge", FirstIntSelector())
    pipeline.add_component("below_10", Threshold(threshold=10))
    pipeline.add_component("accumulator", accumulator)
    pipeline.add_component("add_two", AddFixedValue(add=2))

    pipeline.connect("merge", "below_10.value")
    pipeline.connect("below_10.below", "accumulator.value")
    pipeline.connect("accumulator.value", "merge.inputs")
    pipeline.connect("below_10.above", "add_two.value")

    results = pipeline.run({"merge": {"inputs": 4}})
    assert results == {"add_two": {"result": 18}}
    assert accumulator.state == 16
