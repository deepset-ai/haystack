from typing import Optional, Dict, Any, Set, Callable

import jinja2.runtime
from jinja2 import TemplateSyntaxError, meta
from jinja2.nativetypes import NativeEnvironment
from typing_extensions import TypeAlias

from haystack import component, default_to_dict, default_from_dict
from haystack.utils import serialize_callable, deserialize_callable
from haystack.utils import serialize_type, deserialize_type


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

    ```python
    from haystack import Pipeline, component
    from haystack.components.converters import OutputAdapter

    @component
    class DocumentProducer:
        @component.output_types(documents=dict)
        def run(self):
            return {"documents": [{"content": '{"framework": "Haystack"}'}]}

    pipe = Pipeline()
    pipe.add_component(
        name="output_adapter",
        instance=OutputAdapter(template="{{ documents[0].content | json_loads}}", output_type=str),
    )
    pipe.add_component(name="document_producer", instance=DocumentProducer())
    pipe.connect("document_producer", "output_adapter")
    result = pipe.run(data={})
    assert result["output_adapter"]["output"] == {"framework": "Haystack"}
    ```
    """

    def __init__(self, template: str, output_type: TypeAlias, custom_filters: Optional[Dict[str, Callable]] = None):
        """
        Initializes the OutputAdapter with a set of adaptation rules.
        :param template: A Jinja2 template string that defines how to adapt the output data to the input of the
        downstream component.
        :param output_type: The type of the output data (e.g., str, List[int]).
        :param custom_filters: A dictionary of custom Jinja2 filters to be used in the template.
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
        se_filters = {name: serialize_callable(filter_func) for name, filter_func in self.custom_filters.items()}
        return default_to_dict(
            self, template=self.template, output_type=serialize_type(self.output_type), custom_filters=se_filters
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OutputAdapter":
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
