from typing import Dict, Any, List, Tuple

from pathlib import Path

from canals import node


@node
class StringsToPaths:
    def __init__(self, input: str, output: str):
        # Contract
        self.init_parameters = {"input": input, "output": output}
        self.inputs = [input]
        self.outputs = [output]

    def run(self, name: str, data: List[Tuple[str, Any]], parameters: Dict[str, Any]):
        if not isinstance(data[0][1], list):
            raise ValueError("StringsToPaths only accepts lists.")
        return ({self.outputs[0]: [Path(path) for path in data[0][1]]}, parameters)


def test_strings_to_paths():
    node = StringsToPaths(input="test_in", output="test_out")
    results = node.run(name="test_node", data=[("test_in", ["test_file.txt"])], parameters={})
    assert results == ({"test_out": [Path("test_file.txt")]}, {})
    assert node.init_parameters == {"input": "test_in", "output": "test_out"}
