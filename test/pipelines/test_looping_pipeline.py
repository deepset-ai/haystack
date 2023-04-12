from typing import *
from pathlib import Path
from pprint import pprint

from canals.pipeline import Pipeline
from test.components import Accumulate, AddValue, Below, Merge

import logging

logging.basicConfig(level=logging.DEBUG)


def test_pipeline(tmp_path):
    accumulator = Accumulate(connection="value")

    pipeline = Pipeline(max_loops_allowed=10)
    pipeline.add_component("merge", Merge(inputs=["value", "value"]), input_component=True)
    pipeline.add_component("below_10", Below(threshold=10))
    pipeline.add_component("add_one", AddValue(add=1, input="below"))
    pipeline.add_component("accumulator", accumulator)
    pipeline.add_component("add_two", AddValue(add=2, input="above"))

    pipeline.connect("merge", "below_10")
    pipeline.connect("below_10.below", "add_one")
    pipeline.connect("add_one", "accumulator")
    pipeline.connect("accumulator", "merge")
    pipeline.connect("below_10.above", "add_two")

    try:
        pipeline.draw(tmp_path / "looping_pipeline.png")
    except ImportError:
        logging.warning("pygraphviz not found, pipeline is not being drawn.")

    results = pipeline.run(
        {"value": 3},
    )
    pprint(results)
    print("accumulator: ", accumulator.state)

    assert results == {"value": 12}
    assert accumulator.state == 49


if __name__ == "__main__":
    test_pipeline(Path(__file__).parent)
