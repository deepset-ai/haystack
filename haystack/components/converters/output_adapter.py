import json
from typing import Optional, Dict, Any, Set, Callable

import jinja2.runtime
from jinja2 import TemplateSyntaxError, meta
from jinja2.nativetypes import NativeEnvironment
from haystack.utils.type_serialization import serialize_type, deserialize_type
from haystack import component, default_to_dict, default_from_dict


class OutputAdaptationException(Exception):
    """Exception raised when there is an error during output adaptation."""


@component
class OutputAdapter:
    """
    OutputAdapter in Haystack 2.x pipelines is designed to adapt the output of one component
    to be compatible with the input of another component using Jinja2 template expressions.

    The component configuration requires specifying the adaptation rules. Each rule comprises:
    - 'template': A Jinja2 template string that defines how to adapt the input data.
    - 'output_type': The type of the output data (e.g., str, List[int]).
    - 'custom_filters': A dictionary of custom Jinja2 filters to be used in the template.

    Example configuration:

    ```python
    from haystack.components.converters import OutputAdapter
    adapter = OutputAdapter(template="{{ documents[0].content }}", output_type=str)

    input_data = {"documents": [{"content": "Test content"}]}
    expected_output = {"output": "Test content"}

    assert adapter.run(**input_data) == expected_output
    ```

    In the pipeline setup, the adapter is placed between components that require output/input adaptation.
    The name under which the adapted value is published is `output`. Use this name to connect the OutputAdapter
    to downstream components in the pipeline.

    Example pipeline setup:

    """

    predefined_filters = {
        "json_loads": lambda s: json.loads(s) if isinstance(s, str) else json.loads(str(s)),
    }

    def __init__(self, template: str, output_type: Any, custom_filters: Optional[Dict[str, Callable]] = None):
        """
        Initializes the OutputAdapter with a set of adaptation rules.
        :param template: A Jinja2 template string that defines how to adapt the output data to the input of the
        downstream component.
        :param output_type: The type of the output data (e.g., str, List[int]).
        :param custom_filters: A dictionary of custom Jinja2 filters to be used in the template.
        """
        self.custom_filters = {**(custom_filters or {}), **self.predefined_filters}
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
        Executes the output adaptation logic by applying the specified Jinja template expressions
        to adapt the incoming data to a format suitable for downstream components.

        :param kwargs: A dictionary containing the pipeline variables, which are inputs to the adaptation templates.
        :return: A dictionary containing the adapted outputs, based on the adaptation rules.
        :raises OutputAdaptationException: If there's an error during the adaptation process.
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
        # todo should we serialize the custom filters? And if so, can we do the same as for callback handlers?
        return default_to_dict(self, template=self.template, output_type=serialize_type(self.output_type))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OutputAdapter":
        init_params = data.get("init_parameters", {})
        init_params["output_type"] = deserialize_type(init_params["output_type"])
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
