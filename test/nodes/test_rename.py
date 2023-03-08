from typing import Dict, Any, List, Tuple

from canals import node


@node
class Rename:
    """
    Renames an edge without changing the value.

    Single input, single output node. Doesn't accept parameters.
    Doesn't use stores.

    :param input: the input edge name.
    :param output: the output edge name.
    """

    def __init__(self, input: str, output: str):
        # Contract
        self.init_parameters = {"input": input, "output": output}
        self.inputs = [input]
        self.outputs = [output]

    def run(
        self,
        name: str,
        data: List[Tuple[str, Any]],
        parameters: Dict[str, Any],
        stores: Dict[str, Any],
    ):
        return ({self.outputs[0]: data[0][1]}, parameters)


def test_rename():
    node = Rename(input="test_in", output="test_out")
    results = node.run(name="test_node", data=[("test_in", 10)], parameters={}, stores={})
    assert results == ({"test_out": 10}, {})
    assert node.init_parameters == {"input": "test_in", "output": "test_out"}
