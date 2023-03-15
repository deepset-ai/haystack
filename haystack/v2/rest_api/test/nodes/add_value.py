from typing import Dict, Any, List, Tuple

from canals import node


@node
class AddValue:
    def __init__(self, add: int = 1, input: str = "value", output: str = "value"):
        """
        Adds the value of `add` to the value of the incoming edge.

        Single input, single output node.

        :param add: the value to add. This is also a parameter.
        :param input: name of the input edge
        :param output: name of the output edge
        """
        self.add = add

        self.init_parameters = {"add": add, "input": input, "output": output}
        self.inputs = [input]
        self.outputs = [output]

    def run(self, name: str, data: List[Tuple[str, Any]], parameters: Dict[str, Any]):
        sum = parameters.get(name, {}).get("add", self.add)
        sum += data[0][1]
        return ({self.outputs[0]: sum}, parameters)
