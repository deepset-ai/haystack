# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from haystack.core.pipeline import Pipeline
from haystack.testing.sample_components import Accumulate, AddFixedValue, Threshold, Sum, FirstIntSelector, MergeLoop

import logging

logging.basicConfig(level=logging.DEBUG)


def test_pipeline_fixed():
    accumulator = Accumulate()
    pipeline = Pipeline(max_loops_allowed=10)
    pipeline.add_component("add_zero", AddFixedValue(add=0))
    pipeline.add_component("merge", MergeLoop(expected_type=int, inputs=["in_1", "in_2"]))
    pipeline.add_component("sum", Sum())
    pipeline.add_component("below_10", Threshold(threshold=10))
    pipeline.add_component("add_one", AddFixedValue(add=1))
    pipeline.add_component("counter", accumulator)
    pipeline.add_component("add_two", AddFixedValue(add=2))

    pipeline.connect("add_zero", "merge.in_1")
    pipeline.connect("merge", "below_10.value")
    pipeline.connect("below_10.below", "add_one.value")
    pipeline.connect("add_one.result", "counter.value")
    pipeline.connect("counter.value", "merge.in_2")
    pipeline.connect("below_10.above", "add_two.value")
    pipeline.connect("add_two.result", "sum.values")

    results = pipeline.run({"add_zero": {"value": 8}, "sum": {"values": 2}})
    assert results == {"sum": {"total": 23}}
    assert accumulator.state == 19


def test_pipeline_variadic():
    accumulator = Accumulate()
    pipeline = Pipeline(max_loops_allowed=10)
    pipeline.add_component("add_zero", AddFixedValue(add=0))
    pipeline.add_component("merge", FirstIntSelector())
    pipeline.add_component("sum", Sum())
    pipeline.add_component("below_10", Threshold(threshold=10))
    pipeline.add_component("add_one", AddFixedValue(add=1))
    pipeline.add_component("counter", accumulator)
    pipeline.add_component("add_two", AddFixedValue(add=2))

    pipeline.connect("add_zero", "merge")
    pipeline.connect("merge", "below_10.value")
    pipeline.connect("below_10.below", "add_one.value")
    pipeline.connect("add_one.result", "counter.value")
    pipeline.connect("counter.value", "merge.inputs")
    pipeline.connect("below_10.above", "add_two.value")
    pipeline.connect("add_two.result", "sum.values")

    results = pipeline.run({"add_zero": {"value": 8}, "sum": {"values": 2}})
    assert results == {"sum": {"total": 23}}
    assert accumulator.state == 19
