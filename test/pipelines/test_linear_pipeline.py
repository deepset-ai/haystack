from pathlib import Path
from pprint import pprint

from canals.pipeline import Pipeline
from test.components import AddValue, Double

import logging

logging.basicConfig(level=logging.DEBUG)


def test_pipeline(tmp_path):
    pipeline = Pipeline()
    pipeline.add_component("first_addition", AddValue(add=2))
    pipeline.add_component("second_addition", AddValue())
    pipeline.add_component("double", Double())
    pipeline.connect("first_addition", "double")
    pipeline.connect("double", "second_addition")

    try:
        pipeline.draw(tmp_path / "linear_pipeline.png")
    except ImportError:
        logging.warning("pygraphviz not found, pipeline is not being drawn.")

    results = pipeline.run({"first_addition": {"value": 1}})
    pprint(results)

    assert results == {"second_addition": {"value": 7}}


if __name__ == "__main__":
    test_pipeline(Path(__file__).parent)
