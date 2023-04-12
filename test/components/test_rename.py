from typing import Dict, Any, List, Tuple

from canals import component


@component
class Rename:
    """
    Renames an connection without changing the value.

    Single input, single output component. Doesn't accept parameters.


    :param input: the input connection name.
    :param output: the output connection name.
    """

    def __init__(self, input: str, output: str):
        # Contract
        self.init_parameters = {"input": input, "output": output}
        self.inputs = [input]
        self.outputs = [output]

    def run(self, name: str, data: List[Tuple[str, Any]], parameters: Dict[str, Any]):
        return ({self.outputs[0]: data[0][1]}, parameters)


def test_rename():
    component = Rename(input="test_in", output="test_out")
    results = component.run(name="test_component", data=[("test_in", 10)], parameters={})
    assert results == ({"test_out": 10}, {})
    assert component.init_parameters == {"input": "test_in", "output": "test_out"}
