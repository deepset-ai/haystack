from pathlib import Path
from pprint import pprint

from canals.pipeline import Pipeline
from tests.nodes import AddValue, Double

import logging

logging.basicConfig(level=logging.DEBUG)


def test_pipeline(tmp_path):
    pipeline = Pipeline()
    pipeline.add_node("first_addition", AddValue(add=2))
    pipeline.add_node("second_addition", AddValue(add=1))
    pipeline.add_node("double", Double(input="value"))
    pipeline.connect("first_addition", "double")
    pipeline.connect("double", "second_addition")
    pipeline.draw(tmp_path / "linear_pipeline.png")

    results = pipeline.run({"value": 1})
    pprint(results)

    assert results == {"value": 7}


if __name__ == "__main__":
    test_pipeline(Path(__file__).parent)
