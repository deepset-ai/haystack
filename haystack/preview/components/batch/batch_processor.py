from typing import Type, List, Optional
from collections import OrderedDict

from haystack.preview import component, ComponentError, default_to_dict, default_from_dict, DeserializationError
from haystack.preview.utils.marshalling import marshal_type, unmarshal_type


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
        Component to process batches of items. To use this component effectively, use the `item` output to send the
        batch item to the next component in the pipeline and connect the `current_batch` output of this component
        back to its own `current_batch` input, like so:

        ```python
        pipeline.add_component(name="batch_processor", instance=BatchProcessor(expected_type=Document))
        pipeline.connect("batch_processor.current_batch", "batch_processor.current_batch")
        ```

        :param expected_type: The type of the items in the batch. Sets the input type to its value and the output
            type to a list of this type.
            Keep in mind that only basic types (int, str, etc) and object types are currently supported by the
            `to_dict` and `from_dict` methods. Complex types like `Dict`, `List`, etc... will fail.
        """
        self.expected_type = expected_type
        component.set_input_types(
            self, new_batch=Optional[List[expected_type]], current_batch=Optional[List[expected_type]]
        )
        component.set_output_types(self, item=expected_type, current_batch=List[expected_type])

    def to_dict(self):
        return default_to_dict(self, expected_type=marshal_type(self.expected_type))

    @classmethod
    def from_dict(cls, data):
        if not "expected_type" in data["init_parameters"]:
            raise DeserializationError("The expected_type parameter for BatchCreator is missing.")
        data["init_parameters"]["expected_type"] = unmarshal_type(data["init_parameters"]["expected_type"])
        return default_from_dict(cls, data)

    def run(self, new_batch=None, current_batch=None):
        if new_batch and current_batch:
            raise BatchProcessorError(
                "BatchProcessor received a new batch before the previous one was fully processed."
            )
        if new_batch:
            current_batch = new_batch
        item = current_batch.pop(0) if current_batch else None
        return {"item": item, "current_batch": current_batch or None}
