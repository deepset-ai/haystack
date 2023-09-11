from typing import Type, List, Optional
import sys
import builtins

from haystack.preview import component, ComponentError, default_to_dict, default_from_dict, DeserializationError


class BatchProcessorError(ComponentError):
    ...


@component
class BatchProcessor:
    """
    Component to process batches of items one by one.
    """

    # TODO support more complex input types like `Optional`, `Dict`, `List`, etc...

    def __init__(self, expected_type: Type):
        """
        Component to process batches of items.

        :param expected_type: The type of the items in the batch. Sets the input type to its value and the output
            type to a list of this type.
            Keep in mind that only basic types (int, str, etc) and object types are currently supported by the
            `to_dict` and `from_dict` methods. Complex types like `Dict`, `List`, etc... will fail.
        """
        self.expected_type = expected_type
        component.set_input_types(self, batch=Optional[List[expected_type]], next=Optional[bool])
        component.set_output_types(self, item=expected_type)
        self.batch = None

    def to_dict(self):
        module = self.expected_type.__module__
        if module == "builtins":
            type_name = self.expected_type.__name__
        else:
            type_name = f"{module}.{self.expected_type.__name__}"
        return default_to_dict(self, expected_type=type_name, max_batch_size=self.max_batch_size)

    @classmethod
    def from_dict(cls, data):
        if not "expected_type" in data["init_parameters"]:
            raise DeserializationError("The expected_type parameter for BatchProcessor is missing.")

        if "." not in data["init_parameters"]["expected_type"]:
            expected_type = getattr(builtins, data["init_parameters"]["expected_type"], None)
        else:
            parts = data["init_parameters"]["expected_type"].split(".")
            module_name = ".".join(parts[:-1])
            type_name = parts[-1]
            module = sys.modules.get(module_name, None)
            if not module:
                raise DeserializationError(f"Could not locate the module for the expected_type parameter: {module}")
            expected_type = getattr(module, type_name, None)
            if not expected_type:
                raise DeserializationError(f"Could not locate the expected_type parameter type: {type_name}")

        data["init_parameters"]["expected_type"] = expected_type
        return default_from_dict(cls, data)

    def run(self, batch, next: Optional[bool] = False):
        if batch:
            # The component is receiving a new batch to unroll
            if self.batch:
                raise BatchProcessorError(
                    "BatchProcessor received a new batch before the previous one was fully processed."
                )
            self.batch = batch

        if next:
            # The component is requesting the next item in the batch
            if not self.batch:
                raise BatchProcessorError(
                    "BatchProcessor received a request for the next item before a batch was provided."
                )

            item = self.batch.pop(0)
            return {"item": item}

        return {}
