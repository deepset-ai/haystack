from typing import Dict, Any, List, Tuple, Optional

from pathlib import Path
from pprint import pprint

from canals.pipeline import Pipeline
from canals.nodes import node

import logging

logging.basicConfig(level=logging.DEBUG)


@node
class AddValue:
    """
    Single input, single output node
    """

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
    def __init__(self, input_name: str = "value", output_name: Optional[str] = None):
        # Contract
        self.init_parameters = {"input_name": input_name}
        self.inputs = [input_name]
        if not output_name:
            output_name = input_name
        self.outputs = [output_name]

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


@node
class Enumerate:
    """
    Single input, multi output node, returning the input value on all output edges
    """

    def __init__(self, input_name: str, outputs_count: int):
        self.input_name = input_name
        self.outputs_count = outputs_count
        # Contract
        self.init_parameters = {
            "input_name": input_name,
            "outputs_count": outputs_count,
        }
        self.inputs = [input_name]
        self.outputs = [str(out) for out in range(outputs_count)]

    def run(
        self,
        name: str,
        data: List[Tuple[str, Any]],
        parameters: Dict[str, Any],
        stores: Dict[str, Any],
    ):
        if len(data) != 1:
            raise ValueError("This node accepts a single input.")

        return {output: data[0][1] for output in self.outputs}


def test_pipeline(tmp_path):
    add_one = AddValue(add=1, input_name="value")

    pipeline = Pipeline()
    pipeline.add_node("add_one", add_one)
    pipeline.add_node("enumerate", Enumerate(input_name="value", outputs_count=3))
    pipeline.add_node("add_ten", AddValue(add=10, input_name="0"))
    pipeline.add_node("double", Double(input_name="1", output_name="value"))
    pipeline.add_node("add_three", AddValue(add=3, input_name="2"))
    pipeline.add_node("add_one_again", add_one)

    pipeline.connect(["add_one", "enumerate.1"])
    pipeline.connect(["enumerate.0", "add_ten"])
    pipeline.connect(["enumerate.1", "double"])
    pipeline.connect(["enumerate.2", "add_three", "add_one_again"])
    pipeline.draw(tmp_path / "parallel_branches_pipeline.png")

    results = pipeline.run({"value": 1})
    pprint(results)

    assert results == {
        "add_one_again": [{"value": 6}],
        "add_ten": [{"value": 12}],
        "double": [{"value": 4}],
    }


if __name__ == "__main__":
    test_pipeline(Path(__file__).parent)
