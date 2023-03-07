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
class Double:
    def __init__(self, inputs_name: str = "value"):
        # Contract
        self.init_parameters = {"inputs_name": inputs_name}
        self.inputs = [inputs_name]
        self.outputs = [inputs_name]

    def run(
        self,
        name: str,
        data: List[Tuple[str, Any]],
        parameters: Dict[str, Any],
        stores: Dict[str, Any],
    ):
        for _, value in data:
            value *= 2

        return {self.outputs[0]: value}


def test_pipeline(tmp_path):
    pipeline = Pipeline()
    pipeline.add_node("first_addition", AddValue(add=2))
    pipeline.add_node("second_addition", AddValue(add=1))
    pipeline.add_node("double", Double(inputs_name="value"))
    pipeline.connect(["first_addition", "double", "second_addition"])
    pipeline.draw(tmp_path / "linear_pipeline.png")

    results = pipeline.run({"value": 1})
    pprint(results)

    assert results == {"value": 7}

    pipeline.save(tmp_path / "linear_pipeline.toml")


if __name__ == "__main__":
    test_pipeline(Path(__file__).parent)
