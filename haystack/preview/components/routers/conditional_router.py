import logging
from typing import List, Dict, Any, Set

from jinja2.nativetypes import NativeEnvironment
from jinja2 import meta

from haystack.preview import component

logger = logging.getLogger(__name__)


class NoRouteSelectedException(Exception):
    """Exception raised when no route is selected in ConditionalRouter."""


class RouteConditionException(Exception):
    """Exception raised when there is an error parsing or evaluating the condition expression in ConditionalRouter."""


@component
class ConditionalRouter:
    """
    The ConditionalRouter manages data flow by evaluating route conditions to select the appropriate
    route from a set of specified routes.

    In Haystack 2.x pipelines, utilize a ConditionalRouter by defining a list named 'routes', with
    each list element being a dictionary that represents a single route.

    Each route dictionary must include three keys: 'condition', 'output', and 'output_type'. An
    optional fourth key, 'output_name', is also supported.

    'condition' is a string with a Jinja2 expression that is evaluated to decide if the route is chosen.
    'output' is a Jinja2 expression that determines the output name for the route, and 'output_type'
    specifies the expected output data type (e.g., str, List[int]).

    Example:

        Below, we instantiate a `ConditionalRouter` with two routes. The first route is taken if there are
        fewer than two streams, directing to the `query` output. The second is selected for two or more
        streams, pointing to the `streams` output. Variables `query` and `streams` should be provided in the
        pipeline's `run()` method.

        ```python
        routes = [
            {"condition": "{{streams|length < 2}}", "output": "{{query}}", "output_type": str},
            {"condition": "{{streams|length >= 2}}", "output": "{{streams}}", "output_type": List[ByteStream]}
        ]

        router = ConditionalRouter(routes=routes)
        ```

    Note the use of the router's 'output' field to denote both the output name and the variable value to emit.
    For more complex scenarios, 'output_name' can specify a different output name for the variable defined
    in 'output' expression This flexibility allows more expressive and rich data flow management with the
    ConditionalRouter.

    Here's an example:

    ```python
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
    # When 'streams' has more than 2 items, 'enough_streams' will activate, emitting the list [1, 2, 3]
    kwargs = {"streams": [1, 2, 3], "query": "Haystack"}
    result = router.run(**kwargs)
    assert result == {"enough_streams": [1, 2, 3]}
    ```

    In this scenario, the ConditionalRouter is configured with two routes: the first emits value of 'streams'
    variable to 'enough_streams' output if there are more than two streams; the second routes 'streams' to
    'insufficient_streams' output when there are two or fewer streams.
    """

    def __init__(self, routes: List[Dict]):
        """
        Initializes the ConditionalRouter with a list of routes detailing the conditions for routing.

        :param routes: A list of dictionaries, each defining a route with a boolean condition expression
                       ('condition'), an output ('output'), and the output type as a string representation
                       ('output_type'). An optional 'output_name' can be specified to define a different
                        output name for the variable defined in 'output'.
        """
        self._validate_routes(routes)
        self.routes: List[dict] = routes

        # Create a Jinja native environment to extract variables from the condition templates
        env = NativeEnvironment()

        # Inspect the routes to determine input and output types.
        input_names: Set[str] = set()  # let's just store the name, type will always be Any
        output_types: Dict[str, str] = {}
        for route in routes:
            # Input types must include any variable that needs to be sent in output
            output = route["output"].strip("{}")
            input_names.add(output)
            # Also add any additional variable that might be used within a "condition" expression.
            ast = env.parse(route["condition"])
            input_names.update(meta.find_undeclared_variables(ast))

            # set proper output as well
            output_name = route.get("output_name", output)
            output_types.update({output_name: route["output_type"]})

        component.set_input_types(self, **{var: Any for var in input_names})
        component.set_output_types(self, **output_types)

    def run(self, **kwargs):
        """
        Executes the routing logic by evaluating the specified boolean condition expressions
        for each route in the order they are listed. The method directs the flow
        of data to the output specified in the first route whose expression
        evaluates to True. If no route's expression evaluates to True, an exception
        is raised.

        :param kwargs: A dictionary containing the pipeline variables, which should
                   include all variables used in the "condition" templates.

        :return: A dictionary containing the output and the corresponding result,
             based on the first route whose expression evaluates to True.

        :raises NoRouteSelectedException: If no route's expression evaluates to True.
        """
        # Create a Jinja native environment evaluate the condition templates
        env = NativeEnvironment()

        for route in self.routes:
            try:
                t = env.from_string(route["condition"])
                if t.render(**kwargs):
                    # if optional field output_name is not provided, use mandatory output
                    # but since output is jinja expression, we need to strip the curly braces
                    output = route["output"].strip("{}")
                    output_name = route.get("output_name", output)
                    # value we output is always under the output key
                    output_value = kwargs[output]
                    return {output_name: output_value}
            except Exception as e:
                raise RouteConditionException(f"Error evaluating condition for route '{route}': {e}") from e

        raise NoRouteSelectedException(f"No route fired. Routes: {self.routes}")

    def _validate_routes(self, routes: List[Dict]):
        for route in routes:
            try:
                keys = set(route.keys())
            except AttributeError:
                raise ValueError(f"Route must be a dictionary, got: {route}")

            mandatory_fields = {"condition", "output", "output_type"}
            optional_fields = {"output_name"}
            has_all_mandatory_fields = mandatory_fields.issubset(keys)
            if not has_all_mandatory_fields:
                raise ValueError("Each route must contain 'condition', 'output', and 'output_type' keys.")
            if not keys.issubset(mandatory_fields.union(optional_fields)):
                raise ValueError(
                    f"Route contains invalid keys. Valid keys are: {mandatory_fields.union(optional_fields)}"
                )
            # validate condition and output are valid jinja expressions
            for field in ["condition", "output"]:
                if route[field].count("{{") != 1 or route[field].count("}}") != 1:
                    raise ValueError(f"Route {route} contains invalid field: {field}")
