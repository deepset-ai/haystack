# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import ast
import contextlib
from typing import Any, Callable, Dict, Optional, Set

import jinja2.runtime
from jinja2 import Environment, TemplateSyntaxError, meta
from jinja2.nativetypes import NativeEnvironment
from jinja2.sandbox import SandboxedEnvironment
from typing_extensions import TypeAlias

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.utils import deserialize_callable, deserialize_type, serialize_callable, serialize_type

logger = logging.getLogger(__name__)


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

    def __init__(
        self,
        template: str,
        output_type: TypeAlias,
        custom_filters: Optional[Dict[str, Callable]] = None,
        unsafe: bool = False,
    ):
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
        :param unsafe:
            Enable execution of arbitrary code in the Jinja template.
            This should only be used if you trust the source of the template as it can be lead to remote code execution.
        """
        self.custom_filters = {**(custom_filters or {})}
        input_types: Set[str] = set()

        self._unsafe = unsafe

        if self._unsafe:
            msg = (
                "Unsafe mode is enabled. This allows execution of arbitrary code in the Jinja template. "
                "Use this only if you trust the source of the template."
            )
            logger.warning(msg)
        self._env = (
            NativeEnvironment() if self._unsafe else SandboxedEnvironment(undefined=jinja2.runtime.StrictUndefined)
        )

        try:
            self._env.parse(template)  # Validate template syntax
            self.template = template
        except TemplateSyntaxError as e:
            raise ValueError(f"Invalid Jinja template '{template}': {e}") from e

        for name, filter_func in self.custom_filters.items():
            self._env.filters[name] = filter_func

        # b) extract variables in the template
        route_input_names = self._extract_variables(self._env)
        input_types.update(route_input_names)

        # the env is not needed, discarded automatically
        component.set_input_types(self, **dict.fromkeys(input_types, Any))
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
        for name, filter_func in self.custom_filters.items():
            self._env.filters[name] = filter_func
        adapted_outputs = {}
        try:
            adapted_output_template = self._env.from_string(self.template)
            output_result = adapted_output_template.render(**kwargs)
            if isinstance(output_result, jinja2.runtime.Undefined):
                raise OutputAdaptationException(f"Undefined variable in the template {self.template}; kwargs: {kwargs}")

            # We suppress the exception in case the output is already a string, otherwise
            # we try to evaluate it and would fail.
            # This must be done cause the output could be different literal structures.
            # This doesn't support any user types.
            with contextlib.suppress(Exception):
                if not self._unsafe:
                    output_result = ast.literal_eval(output_result)

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
            self,
            template=self.template,
            output_type=serialize_type(self.output_type),
            custom_filters=se_filters,
            unsafe=self._unsafe,
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

        custom_filters = init_params.get("custom_filters", {})
        if custom_filters:
            init_params["custom_filters"] = {
                name: deserialize_callable(filter_func) if filter_func else None
                for name, filter_func in custom_filters.items()
            }
        return default_from_dict(cls, data)

    def _extract_variables(self, env: Environment) -> Set[str]:
        """
        Extracts all variables from a list of Jinja template strings.

        :param env: A Jinja environment.
        :return: A set of variable names extracted from the template strings.
        """
        ast = env.parse(self.template)
        return meta.find_undeclared_variables(ast)
