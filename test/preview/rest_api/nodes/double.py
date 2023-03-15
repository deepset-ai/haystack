from typing import Dict, Any, List, Tuple

from canals import node


@node
class Double:
    def __init__(self, input: str = "value", output: str = "value"):
        """
        Doubles the value in input.

        Single input single output node. Doesn't take parameters.

        :param input: the name of the input.
        :param output: the name of the output.
        """
        self.init_parameters = {"input": input, "output": output}
        self.inputs = [input]
        self.outputs = [output]

    def run(self, name: str, data: List[Tuple[str, Any]], parameters: Dict[str, Any]):
        for _, value in data:
            value *= 2

        return ({self.outputs[0]: value}, parameters)
