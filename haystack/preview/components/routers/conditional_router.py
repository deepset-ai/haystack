import importlib
import inspect
import logging
import sys
from typing import List, Dict, Any, Set, get_origin

from jinja2 import meta, Environment, TemplateSyntaxError
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
    :type target: Any
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
    ConditionalRouter in Haystack 2.x pipelines is designed to manage data routing based on specific conditions.
    This is achieved by defining a list named 'routes'. Each element in this list is a dictionary representing a
    single route.

    A route dictionary comprises four key elements:
    - 'condition': A Jinja2 string expression that determines if the route is selected.
    - 'output': A Jinja2 expression defining the route's output value.
    - 'output_type': The type of the output data (e.g., str, List[int]).
    - 'output_name': The name under which the output of the route is published. Similar to other Haystack components,
    ConditionalRouter allows specifying output types with the 'output_type' key.

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

    In this example, two routes are configured. The first route sends the 'streams' value to 'enough_streams' if the
    stream count exceeds two. Conversely, the second route directs 'streams' to 'insufficient_streams' when there
    are two or fewer streams.
    """

    def __init__(self, routes: List[Dict]):
        """
        Initializes the ConditionalRouter with a list of routes detailing the conditions for routing.

        :param routes: A list of dictionaries, each defining a route with a boolean condition expression
                       ('condition'), an output value ('output'), the output type ('output_type') and
                       'output_name' that defines output name for the variable defined in 'output'.
        """
        self._validate_routes(routes)
        self.routes: List[dict] = routes

        # Create a Jinja native environment to inspect variables in the condition templates
        env = NativeEnvironment()

        # Inspect the routes to determine input and output types.
        input_types: Set[str] = set()  # let's just store the name, type will always be Any
        output_types: Dict[str, str] = {}

        for route in routes:
            # extract inputs
            route_input_names = self._extract_variables(env, [route["output"], route["condition"]])
            input_types.update(route_input_names)

            # extract outputs
            output_types.update({route["output_name"]: route["output_type"]})

        component.set_input_types(self, **{var: Any for var in input_types})
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
                    # We now evaluate the `output` expression to determine the route output
                    t_output = env.from_string(route["output"])
                    output = t_output.render(**kwargs)
                    # and return the output as a dictionary under the output_name key
                    return {route["output_name"]: output}
            except Exception as e:
                raise RouteConditionException(f"Error evaluating condition for route '{route}': {e}") from e

        raise NoRouteSelectedException(f"No route fired. Routes: {self.routes}")

    def _validate_routes(self, routes: List[Dict]):
        """
        Validates a list of routes.

        :param routes: A list of routes.
        :type routes: List[Dict]
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

    def _extract_variables(self, env: NativeEnvironment, templates: List[str]) -> Set[str]:
        """
        Extracts all variables from a list of Jinja template strings.

        :param env: A Jinja environment.
        :type env: Environment
        :param templates: A list of Jinja template strings.
        :type templates: List[str]
        :return: A set of variable names.
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
        :type env: Environment
        :param template_text: A Jinja template string.
        :type template_text: str
        :return: True if the template is valid, False otherwise.
        """
        try:
            env.parse(template_text)
            return True
        except TemplateSyntaxError:
            return False
