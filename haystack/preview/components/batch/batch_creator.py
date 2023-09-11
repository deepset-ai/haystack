from typing import Optional, List

import sys
import builtins

from haystack.preview import component, default_from_dict, default_to_dict, DeserializationError


@component
class BatchCreator:
    """
    Component to create batches of items. The batch is released when the batch size is reached or when the
    `release_batch` flag is set to True.
    """

    # TODO support types from the typing module

    def __init__(self, expected_type, max_batch_size: Optional[int] = 0):
        """
        Component to create batches of items. The batch is released when the batch size is reached or when the
        `release_batch` flag is set to True.

        :param expected_type: The type of the items in the batch. Sets the input type to its value and the output
            type to a list of this type.

            Keep in mind that only basic types (int, str, etc) and object types are currently supported by the
            `to_dict` and `from_dict` methods. Types from the `typing` module will fail.

        :param max_batch_size: The maximum size of the batch.
        """
        self.expected_type = expected_type
        self.max_batch_size = max_batch_size
        component.set_input_types(self, item=expected_type, release_batch=bool)
        component.set_output_types(self, batch=List[expected_type])
        self.batch: List[expected_type] = []

    def to_dict(self):
        """
        Returns a serializable dictionary representation of the component.
        """
        module = self.expected_type.__module__
        if module == "builtins":
            type_name = self.expected_type.__name__
        else:
            type_name = f"{module}.{self.expected_type.__name__}"
        return default_to_dict(self, expected_type=type_name, max_batch_size=self.max_batch_size)

    @classmethod
    def from_dict(cls, data):
        """
        Reconstructs the component from a serializable dictionary representation.
        """
        if not "expected_type" in data["init_parameters"]:
            raise DeserializationError("The expected_type parameter for BatchCreator is missing.")

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

    def run(self, item, release_batch: bool = False):
        """
        Simply accumulates the inputs items into a list. When the list reaches the max_batch_size or release_batch is
        set to True, outputs the batch and resets its state.

        :param item: The item to add to the batch.
        :param release_batch: If True, the batch is sent in output.
        """
        self.batch.append(item)

        if not release_batch and (self.max_batch_size is not None and len(self.batch) < self.max_batch_size):
            return {}

        output = {"batch": self.batch}
        self.batch = []
        return output
