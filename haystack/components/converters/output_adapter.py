# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Dict, Optional, Set

import jinja2.runtime
from jinja2 import TemplateSyntaxError, meta
from jinja2.nativetypes import NativeEnvironment
from typing_extensions import TypeAlias

from haystack import component, default_from_dict, default_to_dict
from haystack.utils import deserialize_callable, deserialize_type, serialize_callable, serialize_type


class OutputAdaptationException(Exception):
    """Exception raised when there is an error during output adaptation."""


@component
class OutputAdapter:
    """
    Adapts output of a Component using Jinja templates.

    Usage example:
    ```python
    from haystack import Document
    from haystack.components.converters import OutputAdapter

    adapter = OutputAdapter(template="{{ documents[0].content }}", output_type=str)
    documents = [Document(content="Test content"]
    result = adapter.run(documents=documents)

    assert result["output"] == "Test content"
    ```
    """

    def __init__(self, template: str, output_type: TypeAlias, custom_filters: Optional[Dict[str, Callable]] = None):
        """
        Create an OutputAdapter component.

        :param template:
            A Jinja template that defines how to adapt the input data.
            The variables in the template define the input of this instance.
            e.g.
            With this template:
            ```
            {{ documents[0].content }}
            ```
            The Component input will be `documents`.
        :param output_type:
            The type of output this instance will return.
        :param custom_filters:
            A dictionary of custom Jinja filters used in the template.
        """
        self.custom_filters = {**(custom_filters or {})}
        input_types: Set[str] = set()

        # Create a Jinja native environment, we need it to:
        # a) add custom filters to the environment for filter compilation stage
        env = NativeEnvironment()
        try:
            env.parse(template)  # Validate template syntax
            self.template = template
        except TemplateSyntaxError as e:
            raise ValueError(f"Invalid Jinja template '{template}': {e}") from e

        for name, filter_func in self.custom_filters.items():
            env.filters[name] = filter_func

        # b) extract variables in the template
        route_input_names = self._extract_variables(env)
        input_types.update(route_input_names)

        # the env is not needed, discarded automatically
        component.set_input_types(self, **{var: Any for var in input_types})
        component.set_output_types(self, **{"output": output_type})
        self.output_type = output_type

    def run(self, **kwargs):
        """
        Renders the Jinja template with the provided inputs.

        :param kwargs:
            Must contain all variables used in the `template` string.
        :returns:
            A dictionary with the following keys:
            - `output`: Rendered Jinja template.

        :raises OutputAdaptationException: If template rendering fails.
        """
        # check if kwargs are empty
        if not kwargs:
            raise ValueError("No input data provided for output adaptation")
        env = NativeEnvironment()
        for name, filter_func in self.custom_filters.items():
            env.filters[name] = filter_func
        adapted_outputs = {}
        try:
            adapted_output_template = env.from_string(self.template)
            output_result = adapted_output_template.render(**kwargs)
            if isinstance(output_result, jinja2.runtime.Undefined):
                raise OutputAdaptationException(f"Undefined variable in the template {self.template}; kwargs: {kwargs}")

            adapted_outputs["output"] = output_result
        except Exception as e:
            raise OutputAdaptationException(f"Error adapting {self.template} with {kwargs}: {e}") from e
        return adapted_outputs

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        se_filters = {name: serialize_callable(filter_func) for name, filter_func in self.custom_filters.items()}
        return default_to_dict(
            self, template=self.template, output_type=serialize_type(self.output_type), custom_filters=se_filters
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OutputAdapter":
        """
        Deserializes the component from a dictionary.

        :param data:
            The dictionary to deserialize from.
        :returns:
            The deserialized component.
        """
        init_params = data.get("init_parameters", {})
        init_params["output_type"] = deserialize_type(init_params["output_type"])
        for name, filter_func in init_params.get("custom_filters", {}).items():
            init_params["custom_filters"][name] = deserialize_callable(filter_func) if filter_func else None
        return default_from_dict(cls, data)

    def _extract_variables(self, env: NativeEnvironment) -> Set[str]:
        """
        Extracts all variables from a list of Jinja template strings.

        :param env: A Jinja native environment.
        :return: A set of variable names extracted from the template strings.
        """
        variables = set()
        ast = env.parse(self.template)
        variables.update(meta.find_undeclared_variables(ast))
        return variables
