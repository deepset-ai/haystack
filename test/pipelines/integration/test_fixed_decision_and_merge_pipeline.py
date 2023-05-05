import logging
from pathlib import Path
from pprint import pprint

from canals.pipeline import Pipeline
from test.test_components import AddFixedValue, Parity, Double, Subtract

logging.basicConfig(level=logging.DEBUG)


def test_pipeline(tmp_path):
    add_one = AddFixedValue()

    pipeline = Pipeline()
    pipeline.add_component("add_one", add_one)
    pipeline.add_component("parity", Parity())
    pipeline.add_component("add_ten", AddFixedValue(add=10))
    pipeline.add_component("double", Double())
    pipeline.add_component("add_four", AddFixedValue(add=4))
    pipeline.add_component("add_two", add_one)
    pipeline.add_component("diff", Subtract())

    pipeline.connect("add_one", "parity")
    pipeline.connect("parity.even", "add_four.value")
    pipeline.connect("parity.odd", "double")
    pipeline.connect("add_ten", "diff.first_value")
    pipeline.connect("double", "diff.second_value")
    pipeline.connect("parity.odd", "add_ten.value")
    pipeline.connect("add_four", "add_two")

    pipeline.draw(tmp_path / "fixed_decision_and_merge_pipeline.png")

    results = pipeline.run({"add_one": {"value": 1}, "add_two": {"add": 2}})
    pprint(results)
    assert results == {"add_two": {"value": 8}}

    results = pipeline.run({"add_one": {"value": 2}, "add_two": {"add": 2}})
    pprint(results)
    assert results == {"diff": {"difference": 7}}


if __name__ == "__main__":
    test_pipeline(Path(__file__).parent)
