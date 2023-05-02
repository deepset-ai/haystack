from pathlib import Path
from pprint import pprint

from canals.pipeline import Pipeline
from test.components import AddFixedValue, Double

import logging

logging.basicConfig(level=logging.DEBUG)


def test_pipeline(tmp_path):
    pipeline = Pipeline()
    pipeline.add_component("first_addition", AddFixedValue())
    pipeline.add_component("second_addition", AddFixedValue())
    pipeline.add_component("double", Double())
    pipeline.connect("first_addition", "double")
    pipeline.connect("double", "second_addition")

    try:
        pipeline.draw(tmp_path / "linear_pipeline.png")
    except ImportError:
        logging.warning("pygraphviz not found, pipeline is not being drawn.")

    results = pipeline.run({"first_addition": AddFixedValue.Input(value=1)})
    pprint(results)

    assert results == {"second_addition": AddFixedValue.Input(value=5)}


if __name__ == "__main__":
    test_pipeline(Path(__file__).parent)
