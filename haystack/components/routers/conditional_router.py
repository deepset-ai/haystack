# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import ast
import contextlib
from typing import Any, Callable, Dict, List, Optional, Set

from jinja2 import Environment, TemplateSyntaxError, meta
from jinja2.nativetypes import NativeEnvironment
from jinja2.sandbox import SandboxedEnvironment

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.utils import deserialize_callable, deserialize_type, serialize_callable, serialize_type

logger = logging.getLogger(__name__)


class NoRouteSelectedException(Exception):
    """Exception raised when no route is selected in ConditionalRouter."""


class RouteConditionException(Exception):
    """Exception raised when there is an error parsing or evaluating the condition expression in ConditionalRouter."""


@component
class ConditionalRouter:
    """
    Routes data based on specific conditions.

    You define these conditions in a list of dictionaries called `routes`.
    Each dictionary in this list represents a single route. Each route has these four elements:
    - `condition`: A Jinja2 string expression that determines if the route is selected.
    - `output`: A Jinja2 expression defining the route's output value.
    - `output_type`: The type of the output data (for example, `str`, `List[int]`).
    - `output_name`: The name you want to use to publish `output`. This name is used to connect
    the router to other components in the pipeline.

    ### Usage example

    ```python
    from typing import List
    from haystack.components.routers import ConditionalRouter

    routes = [
        {
            "condition": "{{streams|length > 2}}",
            "output": "{{streams}}",
            "output_name": "enough_streams",
            "output_type": List[int],
        },
        {
            "condition": "{{streams|length <= 2}}",
            "output": "{{streams}}",
            "output_name": "insufficient_streams",
            "output_type": List[int],
        },
    ]
    router = ConditionalRouter(routes)
    # When 'streams' has more than 2 items, 'enough_streams' output will activate, emitting the list [1, 2, 3]
    kwargs = {"streams": [1, 2, 3], "query": "Haystack"}
    result = router.run(**kwargs)
    assert result == {"enough_streams": [1, 2, 3]}
    ```

    In this example, we configure two routes. The first route sends the 'streams' value to 'enough_streams' if the
    stream count exceeds two. The second route directs 'streams' to 'insufficient_streams' if there
    are two or fewer streams.

    In the pipeline setup, the Router connects to other components using the output names. For example,
    'enough_streams' might connect to a component that processes streams, while
    'insufficient_streams' might connect to a component that fetches more streams.


    Here is a pipeline that uses `ConditionalRouter` and routes the fetched `ByteStreams` to
    different components depending on the number of streams fetched:

    ```python
    from typing import List
    from haystack import Pipeline
    from haystack.dataclasses import ByteStream
    from haystack.components.routers import ConditionalRouter

    routes = [
        {
            "condition": "{{streams|length > 2}}",
            "output": "{{streams}}",
            "output_name": "enough_streams",
            "output_type": List[ByteStream],
        },
        {
            "condition": "{{streams|length <= 2}}",
            "output": "{{streams}}",
            "output_name": "insufficient_streams",
            "output_type": List[ByteStream],
        },
    ]

    pipe = Pipeline()
    pipe.add_component("router", router)
    ...
    pipe.connect("router.enough_streams", "some_component_a.streams")
    pipe.connect("router.insufficient_streams", "some_component_b.streams_or_some_other_input")
    ...
    ```
    """

    def __init__(self, routes: List[Dict], custom_filters: Optional[Dict[str, Callable]] = None):
        """
        Initializes the `ConditionalRouter` with a list of routes detailing the conditions for routing.

        :param routes: A list of dictionaries, each defining a route.
            Each route has these four elements:
            - `condition`: A Jinja2 string expression that determines if the route is selected.
            - `output`: A Jinja2 expression defining the route's output value.
            - `output_type`: The type of the output data (for example, `str`, `List[int]`).
            - `output_name`: The name you want to use to publish `output`. This name is used to connect
            the router to other components in the pipeline.
        :param custom_filters: A dictionary of custom Jinja2 filters used in the condition expressions.
            For example, passing `{"my_filter": my_filter_fcn}` where:
            - `my_filter` is the name of the custom filter.
            - `my_filter_fcn` is a callable that takes `my_var:str` and returns `my_var[:3]`.
              `{{ my_var|my_filter }}` can then be used inside a route condition expression:
                `"condition": "{{ my_var|my_filter == 'foo' }}"`.
        """
        self._validate_routes(routes)
        self.routes: List[dict] = routes
        self.custom_filters = custom_filters or {}

        # Create a Jinja native environment to inspect variables in the condition templates
        self._env = SandboxedEnvironment()
        self._env.filters.update(self.custom_filters)

        # Inspect the routes to determine input and output types.
        input_types: Set[str] = set()  # let's just store the name, type will always be Any
        output_types: Dict[str, str] = {}

        for route in routes:
            # extract inputs
            route_input_names = self._extract_variables(self._env, [route["output"], route["condition"]])
            input_types.update(route_input_names)

            # extract outputs
            output_types.update({route["output_name"]: route["output_type"]})

        component.set_input_types(self, **{var: Any for var in input_types})
        component.set_output_types(self, **output_types)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        for route in self.routes:
            # output_type needs to be serialized to a string
            route["output_type"] = serialize_type(route["output_type"])
        se_filters = {name: serialize_callable(filter_func) for name, filter_func in self.custom_filters.items()}
        return default_to_dict(self, routes=self.routes, custom_filters=se_filters)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConditionalRouter":
        """
        Deserializes the component from a dictionary.

        :param data:
            The dictionary to deserialize from.
        :returns:
            The deserialized component.
        """
        init_params = data.get("init_parameters", {})
        routes = init_params.get("routes")
        for route in routes:
            # output_type needs to be deserialized from a string to a type
            route["output_type"] = deserialize_type(route["output_type"])
        for name, filter_func in init_params.get("custom_filters", {}).items():
            init_params["custom_filters"][name] = deserialize_callable(filter_func) if filter_func else None
        return default_from_dict(cls, data)

    def run(self, **kwargs):
        """
        Executes the routing logic.

        Executes the routing logic by evaluating the specified boolean condition expressions for each route in the
        order they are listed. The method directs the flow of data to the output specified in the first route whose
        `condition` is True.

        :param kwargs: All variables used in the `condition` expressed in the routes. When the component is used in a
            pipeline, these variables are passed from the previous component's output.

        :returns: A dictionary where the key is the `output_name` of the selected route and the value is the `output`
            of the selected route.

        :raises NoRouteSelectedException: If no `condition' in the routes is `True`.
        :raises RouteConditionException: If there is an error parsing or evaluating the `condition` expression in the
            routes.
        """
        # Create a Jinja native environment to evaluate the condition templates as Python expressions
        for route in self.routes:
            try:
                t = self._env.from_string(route["condition"])
                rendered = t.render(**kwargs)
                if ast.literal_eval(rendered):
                    # We now evaluate the `output` expression to determine the route output
                    t_output = self._env.from_string(route["output"])
                    output = t_output.render(**kwargs)
                    # We suppress the exception in case the output is already a string, otherwise
                    # we try to evaluate it and would fail.
                    # This must be done cause the output could be different literal structures.
                    # This doesn't support any user types.
                    with contextlib.suppress(Exception):
                        output = ast.literal_eval(output)
                    # and return the output as a dictionary under the output_name key
                    return {route["output_name"]: output}
            except Exception as e:
                raise RouteConditionException(f"Error evaluating condition for route '{route}': {e}") from e

        raise NoRouteSelectedException(f"No route fired. Routes: {self.routes}")

    def _validate_routes(self, routes: List[Dict]):
        """
        Validates a list of routes.

        :param routes: A list of routes.
        """
        env = NativeEnvironment()
        for route in routes:
            try:
                keys = set(route.keys())
            except AttributeError:
                raise ValueError(f"Route must be a dictionary, got: {route}")

            mandatory_fields = {"condition", "output", "output_type", "output_name"}
            has_all_mandatory_fields = mandatory_fields.issubset(keys)
            if not has_all_mandatory_fields:
                raise ValueError(
                    f"Route must contain 'condition', 'output', 'output_type' and 'output_name' fields: {route}"
                )
            for field in ["condition", "output"]:
                if not self._validate_template(env, route[field]):
                    raise ValueError(f"Invalid template for field '{field}': {route[field]}")

    def _extract_variables(self, env: SandboxedEnvironment, templates: List[str]) -> Set[str]:
        """
        Extracts all variables from a list of Jinja template strings.

        :param env: A Jinja environment.
        :param templates: A list of Jinja template strings.
        :returns: A set of variable names.
        """
        variables = set()
        for template in templates:
            ast = env.parse(template)
            variables.update(meta.find_undeclared_variables(ast))
        return variables

    def _validate_template(self, env: Environment, template_text: str):
        """
        Validates a template string by parsing it with Jinja.

        :param env: A Jinja environment.
        :param template_text: A Jinja template string.
        :returns: `True` if the template is valid, `False` otherwise.
        """
        try:
            env.parse(template_text)
            return True
        except TemplateSyntaxError:
            return False
