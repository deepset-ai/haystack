import logging
from pathlib import Path
from pprint import pprint

from canals.pipeline import Pipeline
from test.components import AddValue, Even, Double, Sum

logging.basicConfig(level=logging.DEBUG)


def test_pipeline(tmp_path):
    add_one = AddValue()

    pipeline = Pipeline()
    pipeline.add_component("add_one", add_one)
    pipeline.add_component("parity", Even())
    pipeline.add_component("add_ten", AddValue(add=10))
    pipeline.add_component("double", Double())
    pipeline.add_component("add_four", AddValue(add=4))
    pipeline.add_component("add_one_again", add_one)
    pipeline.add_component("sum", Sum())

    pipeline.connect("add_one", "parity")
    pipeline.connect("parity.even", "add_ten")
    pipeline.connect("parity.odd", "double")
    pipeline.connect("add_ten", "sum")
    pipeline.connect("double", "sum")
    pipeline.connect("parity.odd", "add_four")
    pipeline.connect("add_four", "add_one_again")
    pipeline.connect("add_one_again", "sum")

    try:
        pipeline.draw(tmp_path / "decision_and_merge_pipeline.png")
    except ImportError:
        logging.warning("pygraphviz not found, pipeline is not being drawn.")

    results = pipeline.run({"add_one": {"value": 1}})
    pprint(results)
    assert results == {"sum": {"total": 12}}

    results = pipeline.run({"add_one": {"value": 2}})
    pprint(results)
    assert results == {"sum": {"total": 14}}


if __name__ == "__main__":
    test_pipeline(Path(__file__).parent)
