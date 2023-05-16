# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import *
from pathlib import Path
from pprint import pprint

from canals.pipeline import Pipeline
from test.sample_components import Accumulate, AddFixedValue, Threshold, MergeLoop

import logging

logging.basicConfig(level=logging.DEBUG)


def test_pipeline(tmp_path):
    accumulator = Accumulate()
    merge_loop = MergeLoop(expected_type=int)

    pipeline = Pipeline(max_loops_allowed=10)
    pipeline.add_component("merge", merge_loop)
    pipeline.add_component("below_10", Threshold(threshold=10))
    pipeline.add_component("accumulator", accumulator)
    pipeline.add_component("add_two", AddFixedValue(add=2))

    pipeline.connect("merge", "below_10")
    pipeline.connect("below_10.below", "accumulator")
    pipeline.connect("accumulator", "merge")
    pipeline.connect("below_10.above", "add_two.value")

    pipeline.draw(tmp_path / "looping_pipeline.png")

    results = pipeline.run({"merge": merge_loop.input_type(4)})
    pprint(results)
    print("accumulator: ", accumulator.state)

    assert results == {"add_two": AddFixedValue.Output(value=18)}
    assert accumulator.state == 16


if __name__ == "__main__":
    test_pipeline(Path(__file__).parent)
