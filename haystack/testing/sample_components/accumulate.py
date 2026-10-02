# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import Any

from haystack.core.component import component
from haystack.core.errors import ComponentDeserializationError
from haystack.core.serialization import default_to_dict
from haystack.utils.callable_serialization import deserialize_callable, serialize_callable


def _default_function(first: int, second: int) -> int:
    return first + second


@component
class Accumulate:
    """
    Accumulates the value flowing through the connection into an internal attribute.

    The sum function can be customized. Example of how to deal with serialization when some of the parameters
    are not directly serializable.
    """

    def __init__(self, function: Callable | None = None) -> None:
        """
        Class constructor

        :param function:
            the function to use to accumulate the values.
            The function must take exactly two values.
            If it's a callable, it's used as it is.
            If it's a string, the component will look for it in sys.modules and
            import it at need. This is also a parameter.
        """
        self.state = 0
        self.function: Callable = _default_function if function is None else function

    def to_dict(self) -> dict[str, Any]:
        """Converts the component to a dictionary"""
        return default_to_dict(self, function=serialize_callable(self.function))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Accumulate":
        """Loads the component from a dictionary"""
        if "type" not in data:
            raise ComponentDeserializationError("Missing 'type' in component serialization data")
        if data["type"] != f"{cls.__module__}.{cls.__name__}":
            raise ComponentDeserializationError(f"Class '{data['type']}' can't be deserialized as '{cls.__name__}'")

        init_params = data.get("init_parameters", {})

        # Resolve the function through `deserialize_callable` so it passes the deserialization
        # allowlist instead of importing arbitrary handles directly.
        function = init_params.get("function")
        accumulator_function = deserialize_callable(function) if function else None

        return cls(function=accumulator_function)

    @component.output_types(value=int)
    def run(self, value: int):
        """
        Accumulates the value flowing through the connection into an internal attribute.

        The sum function can be customized.
        """
        self.state = self.function(self.state, value)
        return {"value": self.state}
