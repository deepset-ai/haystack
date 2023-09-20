from typing import Type
from haystack.preview import component


@component
class Join:
    """
    Simple component that joins together a group of inputs of the same type. Works with every type that supports
    the + operator for joining, like lists, strings, etc.
    """

    def __init__(self, inputs_count: int, inputs_type: Type):
        """
        :param inputs_count: the number of inputs to expect.
        :param inputs_type: the type of the inputs. Every type that supports the + operator works.
        """
        component.set_input_types({f"input_{i}": inputs_type for i in range(inputs_count)})
        component.set_output_types(output=inputs_type)

    def run(self, **kwargs):
        output = []
        for values in kwargs.values():
            output += values
        return {"output": output}
