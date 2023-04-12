from pathlib import Path
from pprint import pprint

from canals.pipeline import Pipeline
from test.components import AddValue, Sum

import logging

logging.basicConfig(level=logging.DEBUG)


def test_pipeline(tmp_path):

    add_two = AddValue(add=2)
    make_the_sum = Sum(inputs=["value"] * 2)

    pipeline = Pipeline()
    pipeline.add_component("first_addition", add_two)
    pipeline.add_component("second_addition", add_two)
    pipeline.add_component("third_addition", add_two)
    pipeline.add_component("sum", make_the_sum)
    pipeline.add_component("fourth_addition", AddValue(add=1, input="sum"))

    pipeline.connect("first_addition", "second_addition")
    pipeline.connect("second_addition", "sum")
    pipeline.connect("third_addition", "sum")
    pipeline.connect("sum", "fourth_addition")

    try:
        pipeline.draw(tmp_path / "merging_pipeline.png")
    except ImportError:
        logging.warning("pygraphviz not found, pipeline is not being drawn.")

    results = pipeline.run({"value": 1})
    pprint(results)

    assert results == {"value": 9}


if __name__ == "__main__":
    test_pipeline(Path(__file__).parent)
