# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import List, Any, Optional, Dict
import sys

from haystack.core.component import component
from haystack.core.errors import DeserializationError
from haystack.core.serialization import default_to_dict


@component
class MergeLoop:
    def __init__(self, expected_type: Any, inputs: List[str]):
        component.set_input_types(self, **{input_name: Optional[expected_type] for input_name in inputs})
        component.set_output_types(self, value=expected_type)

        if expected_type.__module__ == "builtins":
            self.expected_type = f"builtins.{expected_type.__name__}"
        elif expected_type.__module__ == "typing":
            self.expected_type = str(expected_type)
        else:
            self.expected_type = f"{expected_type.__module__}.{expected_type.__name__}"

        self.inputs = inputs

    def to_dict(self) -> Dict[str, Any]:  # pylint: disable=missing-function-docstring
        return default_to_dict(self, expected_type=self.expected_type, inputs=self.inputs)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MergeLoop":  # pylint: disable=missing-function-docstring
        if "type" not in data:
            raise DeserializationError("Missing 'type' in component serialization data")
        if data["type"] != f"{cls.__module__}.{cls.__name__}":
            raise DeserializationError(f"Class '{data['type']}' can't be deserialized as '{cls.__name__}'")

        init_params = data.get("init_parameters", {})

        if "expected_type" not in init_params:
            raise DeserializationError("Missing 'expected_type' field in 'init_parameters'")

        if "inputs" not in init_params:
            raise DeserializationError("Missing 'inputs' field in 'init_parameters'")

        module = sys.modules[__name__]
        fully_qualified_type_name = init_params["expected_type"]
        if fully_qualified_type_name.startswith("builtins."):
            module = sys.modules["builtins"]
        type_name = fully_qualified_type_name.split(".")[-1]
        try:
            expected_type = getattr(module, type_name)
        except AttributeError as exc:
            raise DeserializationError(
                f"Can't find type '{type_name}', import '{fully_qualified_type_name}' to fix the issue"
            ) from exc

        inputs = init_params["inputs"]

        return cls(expected_type=expected_type, inputs=inputs)

    def run(self, **kwargs):
        """
        :param kwargs: find the first non-None value and return it.
        """
        for value in kwargs.values():
            if value is not None:
                return {"value": value}
        return {}
