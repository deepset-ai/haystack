from pathlib import Path
from pprint import pprint

from canals.pipeline import Pipeline
from test.test_components import AddFixedValue, Remainder, Double

import logging

logging.basicConfig(level=logging.DEBUG)


def test_pipeline(tmp_path):
    add_one = AddFixedValue(add=1)

    pipeline = Pipeline()
    pipeline.add_component("add_one", add_one)
    pipeline.add_component("remainder", Remainder(divisor=3))
    pipeline.add_component("add_ten", AddFixedValue(add=10))
    pipeline.add_component("double", Double())
    pipeline.add_component("add_three", AddFixedValue(add=3))
    pipeline.add_component("add_one_again", add_one)

    pipeline.connect("add_one", "remainder")
    pipeline.connect("remainder.remainder_is_0", "add_ten.value")
    pipeline.connect("remainder.remainder_is_1", "double")
    pipeline.connect("remainder.remainder_is_2", "add_three.value")
    pipeline.connect("add_three", "add_one_again")

    pipeline.draw(tmp_path / "variable_decision_pipeline.png")

    results = pipeline.run({"add_one": {"value": 1}})
    pprint(results)

    assert results == {"add_one_again": {"value": 6}}


if __name__ == "__main__":
    test_pipeline(Path(__file__).parent)
