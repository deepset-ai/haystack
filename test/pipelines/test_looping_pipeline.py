from typing import *
from pathlib import Path
from pprint import pprint

from canals.pipeline import Pipeline
from test.components import Accumulate, AddFixedValue, Threshold, MergeLoop

import logging

logging.basicConfig(level=logging.DEBUG)


def test_pipeline(tmp_path):
    accumulator = Accumulate()

    pipeline = Pipeline(max_loops_allowed=10)
    pipeline.add_component("merge", MergeLoop(expected_type=int))
    pipeline.add_component("below_10", Threshold(threshold=10))
    pipeline.add_component("add_one", AddFixedValue(add=1))
    pipeline.add_component("accumulator", accumulator)
    pipeline.add_component("add_two", AddFixedValue(add=2))

    pipeline.connect("merge", "below_10")
    pipeline.connect("below_10.below", "add_one.value")
    pipeline.connect("add_one", "accumulator")
    pipeline.connect("accumulator", "merge.second_branch")
    pipeline.connect("below_10.above", "add_two.value")

    pipeline.draw(tmp_path / "looping_pipeline.png")

    results = pipeline.run({"merge": {"first_branch": 3}})
    pprint(results)
    print("accumulator: ", accumulator.state)

    assert results == {"value": 12}
    assert accumulator.state == 49


if __name__ == "__main__":
    test_pipeline(Path(__file__).parent)
