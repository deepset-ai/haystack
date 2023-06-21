# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import *
from pathlib import Path
from pprint import pprint

from canals.pipeline import Pipeline
from test.sample_components import Accumulate, AddFixedValue, Threshold, Sum, MergeLoop

import logging

logging.basicConfig(level=logging.DEBUG)


def test_pipeline(tmp_path):
    accumulator = Accumulate()
    merge_loop = MergeLoop(expected_type=int, inputs=["in_1", "in_2"])
    summer = Sum(inputs=["in_1", "in_2"])
    pipeline = Pipeline(max_loops_allowed=10)
    pipeline.add_component("merge", merge_loop)
    pipeline.add_component("sum", summer)
    pipeline.add_component("below_10", Threshold(threshold=10))
    pipeline.add_component("add_one", AddFixedValue(add=1))
    pipeline.add_component("counter", accumulator)
    pipeline.add_component("add_two", AddFixedValue(add=2))

    pipeline.connect("merge", "below_10")
    pipeline.connect("below_10.below", "add_one.value")
    pipeline.connect("add_one", "counter")
    pipeline.connect("counter", "merge.in_1")
    pipeline.connect("below_10.above", "add_two.value")
    pipeline.connect("add_two", "sum.in_1")

    pipeline.draw(tmp_path / "looping_and_merge_pipeline.png")

    results = pipeline.run(
        {"merge": merge_loop.input(in_2=8), "sum": summer.input(in_2=2)},
    )
    pprint(results)
    print("accumulate: ", accumulator.state)

    assert results == {"sum": summer.output(total=23)}
    assert accumulator.state == 19


if __name__ == "__main__":
    test_pipeline(Path(__file__).parent)
