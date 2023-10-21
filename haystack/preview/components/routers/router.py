import ast
import logging
from typing import List, Dict, Any

from haystack.preview import component

logger = logging.getLogger(__name__)


class NoRouteSelectedException(Exception):
    """Exception raised when no route is selected in Router."""

    def __init__(self, routes, routing_variables, message=None):
        self.routes = routes
        self.routing_variables = routing_variables

        super().__init__(
            (
                f"No route fired. Routes: {routes}, " f"Routing Variables: {routing_variables}"
                if message is None
                else message
            )
        )


class ConditionEvaluationError(Exception):
    """Exception raised when there is an error evaluating the condition expression in Router."""

    def __init__(self, route_directive, message=None):
        self.route_directive = route_directive
        super().__init__(f"Error evaluating condition for route: {route_directive}" if message is None else message)


class Router:
    """
    The Router class orchestrates the flow of data by evaluating specified route conditions
    to determine the appropriate route among a set of provided route alternatives.

    To use a Router in Haystack 2.x pipelines we first define a list called routes, where each element
    is a dictionary representing a route.

    Each route dictionary contains three keys: condition, output, and output_type.

    The condition is a string containing a Python expression that will be evaluated to determine if
    this route should be selected. The output is a string specifying the name of the output slot for
    this route, and output_type is a string representation of the expected type of the output.

    After specifying routes list we need to also provide routing variables - variables that will be
    available for use in the condition expressions within the routes as well as the outputs of the router.

    Example:

        In this example, we create a `Router` instance with two routes.
        The first route will be selected if the number of streams is less than 2,
        and will output the `query` variable. The second route will be selected
        if the number of streams is 2 or more, and will output the `streams` variable.
        We also specify the routing variables, which are `query` and `streams`. These
        variables need to be provided in the pipeline `run()` method.

        ```python

            routes = [
                {"condition": "len(streams) < 2", "output": "query", "output_type": "str"},
                {"condition": "len(streams) >= 2", "output": "streams", "output_type": "List[ByteStream]"}
            ]

            router = Router(routes=routes, routing_variables=["query", "streams"])
    """

    def __init__(self, routes: List[Dict], routing_variables: List[str]):
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
        self._validate_routing_variables(routing_variables)
        self.routes = routes
        self.routing_variables = routing_variables
        component.set_input_types(self, **{var: Any for var in routing_variables})

        all_output_types = {}
        for route in routes:
            all_output_types.update({route["output"]: route["output_type"]})
        component.set_output_types(self, **all_output_types)

    def run(self, **kwargs):
        """
        Executes the routing logic by evaluating the specified boolean condition expressions
        for each route in the order they are listed. The method directs the flow
        of data to the output slot specified in the first route whose expression
        evaluates to True. If no route's expression evaluates to True, an exception
        is raised.

        :param kwargs: A dictionary containing the pipeline variables, which should
                   include all variables listed in `routing_variables`.

        :return: A dictionary containing the output slot and the corresponding result,
             based on the first route whose expression evaluates to True.

        :raises NoRouteSelectedException: If no route's expression evaluates to True.
        """
        local_vars = {context_var: kwargs[context_var] for context_var in self.routing_variables}
        for route_directive in self.routes:
            expr_ast = ast.parse(route_directive["condition"], mode="eval")
            compiled_expr = compile(expr_ast, filename="<string>", mode="eval")
            try:
                result = eval(compiled_expr, local_vars)
            except Exception as e:
                raise ConditionEvaluationError(route_directive) from e
            if result:
                output_slot = route_directive["output"]
                return {output_slot: kwargs[output_slot]}
        raise NoRouteSelectedException(routes=self.routes, routing_variables=self.routing_variables)

    def _validate_routes(self, routes: List[Dict]):
        for route in routes:
            if not isinstance(route, dict):
                raise ValueError("Each route must be a dictionary.")
            if "condition" not in route or "output" not in route or "output_type" not in route:
                raise ValueError("Each route must contain 'condition', 'output', and 'output_type' keys.")

    def _validate_routing_variables(self, routing_variables: List[str]):
        for var in routing_variables:
            if not isinstance(var, str):
                raise ValueError("Each routing variable must be a string.")
