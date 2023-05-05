import logging
from pathlib import Path
from pprint import pprint

from canals.pipeline import Pipeline
from test.test_components import AddFixedValue, Remainder, Double, Sum

logging.basicConfig(level=logging.DEBUG)


def test_pipeline(tmp_path):
    add_one = AddFixedValue()

    pipeline = Pipeline()
    pipeline.add_component("add_one", add_one)
    pipeline.add_component("parity", Remainder())
    pipeline.add_component("add_ten", AddFixedValue(add=10))
    pipeline.add_component("double", Double())
    pipeline.add_component("add_four", AddFixedValue(add=4))
    pipeline.add_component("add_one_again", add_one)
    pipeline.add_component("sum", Sum())

    pipeline.connect("add_one", "parity")
    pipeline.connect("parity.remainder_is_0", "add_ten.value")
    pipeline.connect("parity.remainder_is_1", "double")
    pipeline.connect("add_one", "sum")
    pipeline.connect("add_ten", "sum")
    pipeline.connect("double", "sum")
    pipeline.connect("parity.remainder_is_1", "add_four.value")
    pipeline.connect("add_four", "add_one_again")
    pipeline.connect("add_one_again", "sum")

    pipeline.draw(tmp_path / "variable_decision_and_merge_pipeline.png")

    results = pipeline.run({"add_one": {"value": 1}})
    pprint(results)
    assert results == {"sum": {"total": 14}}

    results = pipeline.run({"add_one": {"value": 2}})
    pprint(results)
    assert results == {"sum": {"total": 17}}


if __name__ == "__main__":
    test_pipeline(Path(__file__).parent)
