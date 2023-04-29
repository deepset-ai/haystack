from typing import Dict, Any, List, Tuple, Set

from canals import component


@component
class Distribute:
    """
    Repeats the input value on all outputs.

    Single input, multi output component. Order of outputs is irrelevant.
    Doesn't accept parameters.

    :param input: the name of the input connection.
    :param outputs: the list of the output connections.
    """

    def __init__(self, input: str = "value", outputs: Set[str] = {"first", "second"}):
        self.inputs = [input]
        self.outputs = outputs

    def run(self, name: str, data: List[Tuple[str, Any]], parameters: Dict[str, Any]):
        return ({output: data[0][1] for output in self.outputs}, parameters)


def test_distribute_default():
    component = Distribute()
    results = component.run(name="test_component", data=[("value", 10)], parameters={})
    assert results == ({"first": 10, "second": 10}, {})
    assert component.init_parameters == {}


def test_distribute_init_params():
    component = Distribute(input="test", outputs={"one", "two"})
    results = component.run(name="test_component", data=[("test", 10)], parameters={})
    assert results == ({"one": 10, "two": 10}, {})
    assert component.init_parameters == {"input": "test", "outputs": {"one", "two"}}
