from typing import Type
from haystack.preview import component, default_from_dict, default_to_dict


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
        if inputs_count < 1:
            raise ValueError("inputs_count must be at least 1")
        self.inputs_count = inputs_count
        self.inputs_type = inputs_type
        component.set_input_types(self, **{f"input_{i}": inputs_type for i in range(inputs_count)})
        component.set_output_types(self, output=inputs_type)

    def to_dict(self):
        return default_to_dict(self, inputs_count=self.inputs_count, inputs_type=self.inputs_type)

    @classmethod
    def from_dict(cls, data):
        return default_from_dict(cls, data)

    def run(self, **kwargs):
        """
        Joins together a group of inputs of the same type. Works with every type that supports the + operator,
        like lists, strings, etc.
        """
        if len(kwargs) != self.inputs_count:
            raise ValueError(f"Join expected {self.inputs_count} inputs, but got {len(kwargs)}")

        values = list(kwargs.values())
        output = values[0]
        for values in values[1:]:
            output += values
        return {"output": output}
