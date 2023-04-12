from pathlib import Path
from pprint import pprint

from canals.pipeline import Pipeline
from test.components import AddValue, Double

import logging

logging.basicConfig(level=logging.DEBUG)


def test_pipeline(tmp_path):
    pipeline = Pipeline()
    pipeline.add_component("first_addition", AddValue(add=2))
    pipeline.add_component("second_addition", AddValue(add=1))
    pipeline.add_component("double", Double(input="value"))
    pipeline.connect("first_addition", "double")
    pipeline.connect("double", "second_addition")

    try:
        pipeline.draw(tmp_path / "linear_pipeline.png")
    except ImportError:
        logging.warning("pygraphviz not found, pipeline is not being drawn.")

    results = pipeline.run({"value": 1})
    pprint(results)

    assert results == {"value": 7}


if __name__ == "__main__":
    test_pipeline(Path(__file__).parent)
