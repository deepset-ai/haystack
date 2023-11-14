# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import *
from pathlib import Path
from pprint import pprint

from canals.pipeline import Pipeline
from sample_components import Accumulate, AddFixedValue, Threshold, MergeLoop, FirstIntSelector

import logging

logging.basicConfig(level=logging.DEBUG)


def test_pipeline(tmp_path):
    accumulator = Accumulate()

    pipeline = Pipeline(max_loops_allowed=10)
    pipeline.add_component("add_one", AddFixedValue(add=1))
    pipeline.add_component("merge", MergeLoop(expected_type=int, inputs=["in_1", "in_2"]))
    pipeline.add_component("below_10", Threshold(threshold=10))
    pipeline.add_component("accumulator", accumulator)
    pipeline.add_component("add_two", AddFixedValue(add=2))

    pipeline.connect("add_one.result", "merge.in_1")
    pipeline.connect("merge.value", "below_10.value")
    pipeline.connect("below_10.below", "accumulator.value")
    pipeline.connect("accumulator.value", "merge.in_2")
    pipeline.connect("below_10.above", "add_two.value")

    pipeline.draw(tmp_path / "looping_pipeline.png")

    results = pipeline.run({"add_one": {"value": 3}})
    pprint(results)
    print("accumulator: ", accumulator.state)

    assert results == {"add_two": {"result": 18}}
    assert accumulator.state == 16


def test_pipeline_direct_io_loop(tmp_path):
    accumulator = Accumulate()

    pipeline = Pipeline(max_loops_allowed=10)
    pipeline.add_component("merge", MergeLoop(expected_type=int, inputs=["in_1", "in_2"]))
    pipeline.add_component("below_10", Threshold(threshold=10))
    pipeline.add_component("accumulator", accumulator)

    pipeline.connect("merge.value", "below_10.value")
    pipeline.connect("below_10.below", "accumulator.value")
    pipeline.connect("accumulator.value", "merge.in_2")

    pipeline.draw(tmp_path / "looping_pipeline_direct_io_loop.png")

    results = pipeline.run({"merge": {"in_1": 4}})
    pprint(results)
    print("accumulator: ", accumulator.state)

    assert results == {"below_10": {"above": 16}}
    assert accumulator.state == 16


def test_pipeline_fixed_merger_input(tmp_path):
    accumulator = Accumulate()

    pipeline = Pipeline(max_loops_allowed=10)
    pipeline.add_component("merge", MergeLoop(expected_type=int, inputs=["in_1", "in_2"]))
    pipeline.add_component("below_10", Threshold(threshold=10))
    pipeline.add_component("accumulator", accumulator)
    pipeline.add_component("add_two", AddFixedValue(add=2))

    pipeline.connect("merge.value", "below_10.value")
    pipeline.connect("below_10.below", "accumulator.value")
    pipeline.connect("accumulator.value", "merge.in_2")
    pipeline.connect("below_10.above", "add_two.value")

    pipeline.draw(tmp_path / "looping_pipeline_fixed_merger_input.png")

    results = pipeline.run({"merge": {"in_1": 4}})
    pprint(results)
    print("accumulator: ", accumulator.state)

    assert results == {"add_two": {"result": 18}}
    assert accumulator.state == 16


def test_pipeline_variadic_merger_input(tmp_path):
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

    pipeline.draw(tmp_path / "looping_pipeline_variadic_merger_input.png")

    results = pipeline.run({"merge": {"inputs": 4}})
    pprint(results)
    print("accumulator: ", accumulator.state)

    assert results == {"add_two": {"result": 18}}
    assert accumulator.state == 16


if __name__ == "__main__":
    test_pipeline(Path(__file__).parent)
    test_pipeline_direct_io_loop(Path(__file__).parent)
    test_pipeline_fixed_merger_input(Path(__file__).parent)
    test_pipeline_variadic_merger_input(Path(__file__).parent)
