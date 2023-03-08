from typing import Dict, Any, List, Tuple, Union, Callable, Optional
import sys
import builtins
from importlib import import_module

from canals import node


@node
class Accumulate:
    """
    Accumulates the value flowing through the edge into an internal attribute.

    Example of how to deal with serialization when some of the parameters
    are not directly serializable.

    Stateful, single input, single output node. Does not use stores.

    :param edge: the edge to read the value from.
    :param function: the function to use to accumulate the values.
        The function must take exactly two values.
        If it's a callable, it's used as it is.
        If it's a string, the node will look for it in sys.modules and
        import it at need. This is also a parameter.
    """

    def __init__(self, edge: str, function: Optional[Union[Callable, str]] = None):
        self.state = 0

        if function is None:
            self.function = lambda x, y: x + y
        else:
            self.function = self._load_function(function)

        self.init_parameters = {"edge": edge, "function": self._save_function(function) if function else None}
        self.inputs = [edge]
        self.outputs = [edge]

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

    def run(
        self,
        name: str,
        data: List[Tuple[str, Any]],
        parameters: Dict[str, Any],
        stores: Dict[str, Any],
    ):
        function = parameters.get(name, {}).get("function", self.function)
        self.state = function(self.state, data[0][1])
        return ({data[0][0]: data[0][1]}, parameters)


def test_accumulate_default():
    node = Accumulate(edge="test")
    results = node.run(name="test_node", data=[("test", 10)], parameters={}, stores={})
    assert results == ({"test": 10}, {})
    assert node.state == 10
    assert node.init_parameters == {"edge": "test", "function": None}


def my_subtract(first, second):
    return first - second


def test_accumulate_callable():
    node = Accumulate(edge="test", function=my_subtract)
    results = node.run(name="test_node", data=[("test", 10)], parameters={}, stores={})
    assert results == ({"test": 10}, {})
    assert node.state == -10
    assert node.init_parameters == {"edge": "test", "function": "test.nodes.test_accumulate.my_subtract"}


def test_accumulate_string():
    node = Accumulate(edge="test", function="test.nodes.test_accumulate.my_subtract")
    results = node.run(name="test_node", data=[("test", 10)], parameters={}, stores={})
    assert results == ({"test": 10}, {})
    assert node.state == -10
    assert node.init_parameters == {"edge": "test", "function": "test.nodes.test_accumulate.my_subtract"}
