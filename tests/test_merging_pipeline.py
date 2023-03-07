from typing import Dict, Any, List, Tuple

from pathlib import Path
from pprint import pprint

from canals.pipeline import Pipeline
from canals.nodes import *
from canals.nodes import node

import logging

logging.basicConfig(level=logging.DEBUG)


@node
class AddValue:
    def __init__(self, add: int = 1, input_name: str = "value", output_name: str = "value"):
        self.add = add

        # Contract
        self.init_parameters = {"add": add}
        self.inputs = [input_name]
        self.outputs = [output_name]

    def run(
        self,
        name: str,
        data: List[Tuple[str, Any]],
        parameters: Dict[str, Any],
        stores: Dict[str, Any],
    ):
        for _, value in data:
            value += self.add

        return {"value": value}


@node
class Sum:
    def __init__(self, inputs_name: str = "value", inputs_count: int = 2):
        # Contract
        self.init_parameters = {
            "inputs_count": inputs_count,
            "inputs_name": inputs_name,
        }
        self.inputs = [inputs_name] * inputs_count
        self.outputs = ["sum"]

    def run(
        self,
        name: str,
        data: List[Tuple[str, Any]],
        parameters: Dict[str, Any],
        stores: Dict[str, Any],
    ):
        sum = 0
        for _, value in data:
            sum += value

        return {"sum": sum}


def test_pipeline(tmp_path):

    add_two = AddValue(add=2)
    make_the_sum = Sum(inputs_count=2, inputs_name="value")

    pipeline = Pipeline()
    pipeline.add_node("first_addition", add_two)
    pipeline.add_node("second_addition", add_two)
    pipeline.add_node("third_addition", add_two)
    pipeline.add_node("sum", make_the_sum)
    pipeline.add_node("fourth_addition", AddValue(add=1, input_name="sum"))
    pipeline.connect(["first_addition", "second_addition", "sum"])

    # pipeline.connect(["first_addition.edge", "sum"])  # This fails: sum has a free slot, but expects an edge called 'value'

    pipeline.connect(["third_addition", "sum", "fourth_addition"])

    # pipeline.connect(["first_addition", "sum"])  # This fails: sum expects exactly two inputs

    pipeline.draw(tmp_path / "merging_pipeline.png")

    results = pipeline.run({"value": 1})
    pprint(results)

    assert results == {"value": 9}


if __name__ == "__main__":
    test_pipeline(Path(__file__).parent)
