from typing import *
from pathlib import Path
from pprint import pprint

from canals.pipeline import Pipeline
from tests.nodes import Accumulate, AddValue, Below, Merge, Sum

import logging

logging.basicConfig(level=logging.DEBUG)


def test_pipeline(tmp_path):
    accumulator = Accumulate(edge="value")

    pipeline = Pipeline(max_loops_allowed=10)
    pipeline.add_node("merge", Merge(), input_node=True)
    pipeline.add_node("sum", Sum(inputs=["value", "value"]), input_node=True)
    pipeline.add_node("below_10", Below(threshold=10))
    pipeline.add_node("add_one", AddValue(add=1, input="below"))
    pipeline.add_node("counter", accumulator)
    pipeline.add_node("add_two", AddValue(add=2, input="above"))

    pipeline.connect("merge", "below_10")
    pipeline.connect("below_10.below", "add_one")
    pipeline.connect("add_one", "counter")
    pipeline.connect("counter", "merge")
    pipeline.connect("below_10.above", "add_two")
    pipeline.connect("add_two", "sum")

    pipeline.draw(tmp_path / "looping_and_merge_pipeline.png")

    results = pipeline.run(
        {"value": 3},
    )
    pprint(results)
    print("accumulate: ", accumulator.state)

    assert results == {"sum": 15}
    assert accumulator.state == 7


if __name__ == "__main__":
    test_pipeline(Path(__file__).parent)
