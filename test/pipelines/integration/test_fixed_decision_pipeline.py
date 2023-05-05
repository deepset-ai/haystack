from pathlib import Path
from pprint import pprint

from canals.pipeline import Pipeline
from test.test_components import AddFixedValue, Parity, Double

import logging

logging.basicConfig(level=logging.DEBUG)


def test_pipeline(tmp_path):
    add_one = AddFixedValue(add=1)

    pipeline = Pipeline()
    pipeline.add_component("add_one", add_one)
    pipeline.add_component("parity", Parity())
    pipeline.add_component("add_ten", AddFixedValue(add=10))
    pipeline.add_component("double", Double())
    pipeline.add_component("add_three", AddFixedValue(add=3))

    pipeline.connect("add_one", "parity")
    pipeline.connect("parity.even", "add_ten.value")
    pipeline.connect("parity.odd", "double.value")
    pipeline.connect("add_ten", "add_three")

    pipeline.draw(tmp_path / "fixed_decision_pipeline.png")

    results = pipeline.run({"add_one": {"value": 1}})
    pprint(results)
    assert results == {"add_three": {"value": 15}}

    results = pipeline.run({"add_one": {"value": 2}})
    pprint(results)
    assert results == {"double": {"value": 6}}


if __name__ == "__main__":
    test_pipeline(Path(__file__).parent)
