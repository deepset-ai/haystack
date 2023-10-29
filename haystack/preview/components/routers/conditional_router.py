import logging
from typing import List, Dict, Any, Set

from jinja2.nativetypes import NativeEnvironment
from jinja2 import meta

from haystack.preview import component

logger = logging.getLogger(__name__)


class NoRouteSelectedException(Exception):
    """Exception raised when no route is selected in Router."""


class RouteConditionException(Exception):
    """Exception raised when there is an error parsing or evaluating the condition expression in Router."""


@component
class ConditionalRouter:
    """
    The Router class orchestrates the flow of data by evaluating specified route conditions
    to determine the appropriate route among a set of provided route alternatives.

    To use a Router in Haystack 2.x pipelines we first define a list called routes, where each element
    is a dictionary representing a route.

    Each route dictionary contains three keys: condition, output, and output_type.

    The condition is a string containing a Jinja2 boolean expression that will be evaluated to determine if
    this route should be selected. The output is a string specifying the name of the output slot for
    this route, and output_type is a string representation of the expected type of the output.

    Example:

        In this example, we create a `Router` instance with two routes.
        The first route will be selected if the number of streams is less than 2,
        and will output the `query` variable. The second route will be selected
        if the number of streams is 2 or more, and will output the `streams` variable.
        These variables need to be provided in the pipeline `run()` method.

        ```python

            routes = [
                {"condition": "{{streams|length < 2}}", "output": "query", "output_type": "str"},
                {"condition": "{{streams|length < 2}}", "output": "streams", "output_type": "List[ByteStream]"}
            ]

            router = Router(routes=routes)
        ```
    """

    def __init__(self, routes: List[Dict]):
        """
        Initialize the Router with a list of routes and the routing variables.

        :param routes: A list of dictionaries, each representing a route with a
                       boolean condition expression (`condition`), an output slot (`output`),
                       and the output type as a string representation (`output_type`).

        :param routing_variables: A list of additional pipeline variables that are
                       used in the boolean condition expressions or as outputs of the router.
                       These variables should be provided by either the pipeline `run()`
                       method or by a previous component to the router in the pipeline.

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
            input_names.add(route["output"])
            # Also add any additional variable that might be used within a "condition" expression.
            ast = env.parse(route["condition"])
            input_names.update(meta.find_undeclared_variables(ast))

            output_types.update({route["output"]: route["output_type"]})

        component.set_input_types(self, **{var: Any for var in input_names})
        component.set_output_types(self, **output_types)

    def run(self, **kwargs):
        """
        Executes the routing logic by evaluating the specified boolean condition expressions
        for each route in the order they are listed. The method directs the flow
        of data to the output slot specified in the first route whose expression
        evaluates to True. If no route's expression evaluates to True, an exception
        is raised.

        :param kwargs: A dictionary containing the pipeline variables, which should
                   include all variables used in the "condition" templates.

        :return: A dictionary containing the output slot and the corresponding result,
             based on the first route whose expression evaluates to True.

        :raises NoRouteSelectedException: If no route's expression evaluates to True.
        """
        # Create a Jinja native environment evaluate the condition templates
        env = NativeEnvironment()

        for route in self.routes:
            try:
                t = env.from_string(route["condition"])
                if t.render(**kwargs):
                    output_slot = route["output"]
                    return {output_slot: kwargs[output_slot]}
            except Exception as e:
                raise RouteConditionException(f"Error evaluating condition for route '{route}': {e}") from e

        raise NoRouteSelectedException(f"No route fired. Routes: {self.routes}")

    def _validate_routes(self, routes: List[Dict]):
        for route in routes:
            try:
                keys = set(route.keys())
            except AttributeError:
                raise ValueError(f"Route must be a dictionary, got: {route}")

            if not {"condition", "output", "output_type"}.issubset(keys):
                raise ValueError("Each route must contain 'condition', 'output', and 'output_type' keys.")
