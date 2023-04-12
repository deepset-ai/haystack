from typing import Dict, Any, List, Tuple

from canals import component


@component
class Subtract:
    def __init__(self, first_input: str, second_input: str, output: str = "diff"):
        """
        Subtracts the second connection's value from the first.

        Double input, single output component. Order of input connections is critical.
        Doesn't have parameters.

        :param first_input: name of the connection carrying the value to subtract from.
        :param second_input: name of the connection carrying the value to subtract.
        :param output: name of the output connection.
        """
        self.init_parameters = {"first_input": first_input, "second_input": second_input, "output": output}
        self.inputs = [first_input, second_input]
        self.outputs = [output]

    def run(self, name: str, data: List[Tuple[str, Any]], parameters: Dict[str, Any]):
        first_value = [value for name, value in data if name == self.inputs[0]][0]
        second_value = [value for name, value in data if name == self.inputs[1]][0]

        return ({"diff": first_value - second_value}, parameters)


def test_subtract():
    component = Subtract(first_input="first", second_input="second", output="diff")
    results = component.run(name="test_component", data=[("second", 7), ("first", 10)], parameters={})
    assert results == ({"diff": 3}, {})
    assert component.init_parameters == {"first_input": "first", "second_input": "second", "output": "diff"}
