import importlib
import inspect
import logging
import sys
from typing import List, Dict, Any, Set, get_origin

from jinja2 import meta
from jinja2.nativetypes import NativeEnvironment

from haystack.preview import component, default_from_dict, default_to_dict, DeserializationError

logger = logging.getLogger(__name__)


class NoRouteSelectedException(Exception):
    """Exception raised when no route is selected in ConditionalRouter."""


class RouteConditionException(Exception):
    """Exception raised when there is an error parsing or evaluating the condition expression in ConditionalRouter."""


def serialize_type(target: Any) -> str:
    """
    Serializes a type or an instance to its string representation including the module name.

    This function handles types, instances of types, and special typing objects.
    It assumes that non-typing objects will have a '__name__' attribute and raises
    an error if a type cannot be serialized.

    :param target: The object to serialize, which can be an instance or a type.
    :return: The string representation of the type.
    :raises ValueError: If the type cannot be serialized.
    """
    # If the target is a string and contains a dot, treat it as an already serialized type
    if isinstance(target, str) and "." in target:
        return target

    # Determine if the target is a type or an instance of a typing object
    is_type_or_typing = isinstance(target, type) or bool(get_origin(target))
    type_obj = target if is_type_or_typing else type(target)
    module = inspect.getmodule(type_obj)

    # Attempt to extract the type name
    if hasattr(type_obj, "__name__"):
        type_name = type_obj.__name__
    else:
        type_repr = repr(type_obj)
        if type_repr.startswith("typing."):
            type_name = type_repr.split("[", 1)[0][len("typing.") :]
        else:
            # If type cannot be serialized, raise an error
            raise ValueError(f"Could not serialize type: {type_repr}")

    # Construct the full path with module name if available
    if module and hasattr(module, "__name__"):
        full_path = f"{module.__name__}.{type_name}"
    else:
        full_path = type_name

    return full_path


def deserialize_type(type_str: str) -> Any:
    """
    Deserializes a type given its full import path as a string.

    This function will dynamically import the module if it's not already imported,
    and then retrieve the type object from it.

    :param type_str: The string representation of the type's full import path.
    :return: The deserialized type object.
    :raises DeserializationError: If the type cannot be deserialized due to missing module or type.
    """
    parts = type_str.split(".")
    module_name = ".".join(parts[:-1])
    type_name = parts[-1]

    # Try to get the module from sys.modules or import it if necessary
    module = sys.modules.get(module_name)
    if not module:
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            raise DeserializationError(f"Could not import the module: {module_name}") from e

    # Attempt to find the type in the module
    deserialized_type = getattr(module, type_name, None)
    if not deserialized_type:
        raise DeserializationError(f"Could not locate the type: {type_name} in the module: {module_name}")

    return deserialized_type


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

        # Create a Jinja native environment to inspect variables in the condition templates
        env = NativeEnvironment()

        # Inspect the routes to determine input and output types.
        input_names: Set[str] = set()  # let's just store the name, type will always be Any
        output_types: Dict[str, str] = {}
        for route in routes:
            # Start with the "output" expression
            output_ast = env.parse(route["output"])
            output_vars = meta.find_undeclared_variables(output_ast)
            if len(output_vars) != 1 and "output_name" not in route:
                error_message = (
                    f"The 'output' field in the route {route} uses {len(output_vars)} variables: {output_vars}. "
                    "Specify an 'output_name' in the route if use of multiple variables is intentional."
                    if output_vars
                    else f"The 'output' field in the route {route} contains a constant. "
                    "Specify an 'output_name' in the route if use of a constant is intentional."
                )
                raise RouteConditionException(error_message)

            input_names.update(output_vars)

            # Also add any additional variable that might be used within a "condition" expression.
            ast = env.parse(route["condition"])
            input_names.update(meta.find_undeclared_variables(ast))

            # output_vars is a set guaranteed to have 0 or 1 elements
            output_var = output_vars.pop() if output_vars else None
            output_name = route.get("output_name", output_var)
            # set proper output as well
            output_types.update({output_name: route["output_type"]})

        component.set_input_types(self, **{var: Any for var in input_names})
        component.set_output_types(self, **output_types)

    def to_dict(self) -> Dict[str, Any]:
        for route in self.routes:
            # output_type needs to be serialized to a string
            route["output_type"] = serialize_type(route["output_type"])

        return default_to_dict(self, routes=self.routes)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConditionalRouter":
        init_params = data.get("init_parameters", {})
        routes = init_params.get("routes")
        for route in routes:
            # output_type needs to be deserialized from a string to a type
            route["output_type"] = deserialize_type(route["output_type"])
        return default_from_dict(cls, data)

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
        # Create a Jinja native environment to evaluate the condition templates as Python expressions
        env = NativeEnvironment()

        for route in self.routes:
            try:
                t = env.from_string(route["condition"])
                if t.render(**kwargs):
                    # We now evaluate the `output` expression to determine what will be this
                    # component's output.
                    t_output = env.from_string(route["output"])
                    output = t_output.render(**kwargs)

                    # In case `output_name` was not specified, we use the name of the variable contained
                    # in `output` to address the appropriate output socket for this component.
                    output_vars = meta.find_undeclared_variables(env.parse(route["output"]))
                    output_var = output_vars.pop() if output_vars else None
                    output_name = route.get("output_name", output_var)
                    return {output_name: output}
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
            # validate condition is jinja expressions
            field = "condition"
            if route[field].count("{{") != 1 or route[field].count("}}") != 1:
                raise ValueError(f"Route {route} contains invalid jinja expression in 'condition' field.")
