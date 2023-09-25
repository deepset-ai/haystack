from typing import Type
from haystack.preview import component, default_from_dict, default_to_dict, DeserializationError, ComponentError
from haystack.preview.utils import marshal_type, unmarshal_type


@component
class Join:
    """
    A simple component that joins together a group of inputs of the same type. Works with every type that supports
    the + operator for joining, such as lists, strings, etc.
    """

    def __init__(self, inputs_count: int, inputs_type: Type):
        """
        :param inputs_count: The number of inputs to expect.
        :param inputs_type: The type of the inputs. Every type that supports the + operator works.
        """
        if inputs_count < 1:
            raise ValueError("inputs_count must be at least 1")
        self.inputs_count = inputs_count
        self.inputs_type = inputs_type
        component.set_input_types(self, **{f"input_{i}": inputs_type for i in range(inputs_count)})
        component.set_output_types(self, output=inputs_type)

    def to_dict(self):
        return default_to_dict(self, inputs_count=self.inputs_count, inputs_type=marshal_type(self.inputs_type))

    @classmethod
    def from_dict(cls, data):
        if not "inputs_type" in data["init_parameters"]:
            raise DeserializationError("The inputs_type parameter for Join is missing.")
        data["init_parameters"]["inputs_type"] = unmarshal_type(data["init_parameters"]["inputs_type"])
        return default_from_dict(cls, data)

    def run(self, **kwargs):
        """
        Joins together a group of inputs of the same type. Works with every type that supports the + operator,
        such as lists, strings, etc.
        """
        if len(kwargs) != self.inputs_count:
            raise ValueError(f"Join expected {self.inputs_count} inputs, but got {len(kwargs)}")

        values = list(kwargs.values())
        output = values[0]
        try:
            for values in values[1:]:
                output += values
        except TypeError:
            raise ComponentError(
                f"Join expected inputs of a type that supports the + operator, but got: {[type(v) for v in values]}"
            )
        return {"output": output}
