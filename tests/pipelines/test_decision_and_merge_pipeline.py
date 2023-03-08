import logging
from pathlib import Path
from pprint import pprint

from canals.pipeline import Pipeline
from tests.nodes import AddValue, Remainder, Double, Sum

logging.basicConfig(level=logging.DEBUG)


def test_pipeline(tmp_path):
    add_one = AddValue(add=1, input="value")

    pipeline = Pipeline()
    pipeline.add_node("add_one", add_one)
    pipeline.add_node("remainder", Remainder(input="value", divisor=3))
    pipeline.add_node("add_ten", AddValue(add=10, input="0"))
    pipeline.add_node("double", Double(input="1", output="value"))
    pipeline.add_node("add_three", AddValue(add=3, input="2"))
    pipeline.add_node("add_one_again", add_one)
    pipeline.add_node("sum", Sum(inputs=["value"] * 4), input_node=True)

    pipeline.connect("add_one", "remainder")
    pipeline.connect("remainder.0", "add_ten")
    pipeline.connect("remainder.1", "double")
    pipeline.connect("add_ten", "sum")
    pipeline.connect("double", "sum")
    pipeline.connect("remainder.2", "add_three")
    pipeline.connect("add_three", "add_one_again")
    pipeline.connect("add_one_again", "sum")

    try:
        pipeline.draw(tmp_path / "decision_and_merge_pipeline.png")
    except ImportError:
        logging.warning("pygraphviz not found, pipeline is not being drawn.")

    results = pipeline.run({"value": 1})
    pprint(results)

    assert results == {"sum": 7}


if __name__ == "__main__":
    test_pipeline(Path(__file__).parent)
