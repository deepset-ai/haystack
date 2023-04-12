from typing import Dict, Any, List, Tuple, Union, Callable, Optional
import sys
import builtins
from importlib import import_module

from canals import component


@component
class Accumulate:
    """
    Accumulates the value flowing through the connection into an internal attribute.

    Example of how to deal with serialization when some of the parameters
    are not directly serializable.

    Stateful, single input, single output component.

    :param connection: the connection to read the value from.
    :param function: the function to use to accumulate the values.
        The function must take exactly two values.
        If it's a callable, it's used as it is.
        If it's a string, the component will look for it in sys.modules and
        import it at need. This is also a parameter.
    """

    def __init__(self, connection: str, function: Optional[Union[Callable, str]] = None):
        self.state = 0

        if function is None:
            self.function = lambda x, y: x + y
        else:
            self.function = self._load_function(function)

        self.init_parameters = {
            "connection": connection,
            "function": self._save_function(function) if function else None,
        }
        self.inputs = [connection]
        self.outputs = [connection]

    def _load_function(self, function: Union[Callable, str]):
        """
        Loads the function by trying to import it.
        """
        if not isinstance(function, str):
            return function

        parts = function.split(".")
        module_name = ".".join(parts[:-1])
        function_name = parts[-1]

        module = import_module(module_name)
        return getattr(module, function_name)

    def _save_function(self, function: Union[Callable, str]):
        """
        Saves the function by returning its import path to be used with `_load_function`
        (which uses `import_module` internally).
        """
        if isinstance(function, str):
            return function
        module = sys.modules.get(function.__module__)
        if not module:
            raise ValueError("Could not locate the import module.")
        if module == builtins:
            return function.__name__
        return f"{module.__name__}.{function.__name__}"

    def run(self, name: str, data: List[Tuple[str, Any]], parameters: Dict[str, Any]):
        function = parameters.get(name, {}).get("function", self.function)
        self.state = function(self.state, data[0][1])
        return ({data[0][0]: data[0][1]}, parameters)


def test_accumulate_default():
    component = Accumulate(connection="test")
    results = component.run(name="test_component", data=[("test", 10)], parameters={})
    assert results == ({"test": 10}, {})
    assert component.state == 10
    assert component.init_parameters == {"connection": "test", "function": None}


def my_subtract(first, second):
    return first - second


def test_accumulate_callable():
    component = Accumulate(connection="test", function=my_subtract)
    results = component.run(name="test_component", data=[("test", 10)], parameters={})
    assert results == ({"test": 10}, {})
    assert component.state == -10
    assert component.init_parameters == {
        "connection": "test",
        "function": "test.components.test_accumulate.my_subtract",
    }


def test_accumulate_string():
    component = Accumulate(connection="test", function="test.components.test_accumulate.my_subtract")
    results = component.run(name="test_component", data=[("test", 10)], parameters={})
    assert results == ({"test": 10}, {})
    assert component.state == -10
    assert component.init_parameters == {
        "connection": "test",
        "function": "test.components.test_accumulate.my_subtract",
    }
