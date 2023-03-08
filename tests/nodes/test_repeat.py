from typing import Dict, Any, List, Tuple, Set

from canals import node


@node
class Repeat:
    """
    Repeats the input value on all outputs.

    Single input, multi output node. Order of outputs is irrelevant.
    Doesn't accept parameters. Doesn't use stores.

    :param input: the name of the input edge.
    :param outputs: the list of the output edges.
    """

    def __init__(self, input: str = "value", outputs: Set[str] = {"first", "second"}):
        self.init_parameters = {
            "input": input,
            "outputs": outputs,
        }
        self.inputs = [input]
        self.outputs = outputs

    def run(
        self,
        name: str,
        data: List[Tuple[str, Any]],
        parameters: Dict[str, Any],
        stores: Dict[str, Any],
    ):
        return ({output: data[0][1] for output in self.outputs}, parameters)


def test_repeat_default():
    node = Repeat()
    results = node.run(name="test_node", data=[("value", 10)], parameters={}, stores={})
    assert results == ({"first": 10, "second": 10}, {})
    assert node.init_parameters == {"input": "value", "outputs": {"first", "second"}}


def test_repeat_init_params():
    node = Repeat(input="test", outputs={"one", "two"})
    results = node.run(name="test_node", data=[("test", 10)], parameters={}, stores={})
    assert results == ({"one": 10, "two": 10}, {})
    assert node.init_parameters == {"input": "test", "outputs": {"one", "two"}}
