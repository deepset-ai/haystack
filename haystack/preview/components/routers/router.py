import ast
import logging
from typing import List, Dict, Any

from haystack.preview import component

logger = logging.getLogger(__name__)


@component
class Router:
    def __init__(self, routes: Dict[str, dict], input_context_vars: List[str]):
        """
        Initialize the Router.

        In routes, the key is the component pipeline registration name and the value is boolean expression.

        The input_context_vars are additional pipeline variables that are used in the boolean expressions or
        outputs of the router. These variables should be provided by pipeline run() method.
        """
        self.routes = routes
        self.input_context_vars = input_context_vars
        component.set_input_types(self, **{var: Any for var in input_context_vars})

        all_output_types = {}
        for route in routes.values():
            all_output_types.update({route["output"]: route["output_type"]})
        component.set_output_types(self, **all_output_types)

    def run(self, **kwargs):
        # some routing execution logic
        # resolve the firing boolean expression and return output slots and corresponding result
        local_vars = {context_var: kwargs[context_var] for context_var in self.input_context_vars}
        for route_directive in self.routes.values():
            if route_directive["expression"] and route_directive["output"]:
                expr_ast = ast.parse(route_directive["expression"], mode="eval")
                compiled_expr = compile(expr_ast, filename="<string>", mode="eval")
                result = eval(compiled_expr, local_vars)
                if result:
                    output_slot = route_directive["output"]
                    return {output_slot: kwargs[output_slot]}
        raise Exception("No route fired")
