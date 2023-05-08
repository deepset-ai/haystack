from typing import Union, Callable, Optional
import sys
import builtins
from importlib import import_module
from dataclasses import dataclass

import pytest
from canals import component
from canals.testing import BaseTestComponent


@component
class Accumulate:
    """
    Accumulates the value flowing through the connection into an internal attribute.
    The sum function can be customized.

    Example of how to deal with serialization when some of the parameters
    are not directly serializable.

    Stateful, single input, single output component.
    """

    @dataclass
    class Output:
        value: int

    def __init__(self, function: Optional[Union[Callable, str]] = None):
        """
        :param function: the function to use to accumulate the values.
            The function must take exactly two values.
            If it's a callable, it's used as it is.
            If it's a string, the component will look for it in sys.modules and
            import it at need. This is also a parameter.
        """
        self.state = 0

        if function is None:
            self.function = lambda x, y: x + y
        else:
            self.function = self._load_function(function)
            # 'function' is not serializable by default, so we serialize it manually.
            self._init_parameters = {"function": self._save_function(function)}

    def run(self, value: int) -> Output:
        self.state = self.function(self.state, value)
        return Accumulate.Output(value=self.state)

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


def my_subtract(first, second):
    return first - second


class TestAccumulate(BaseTestComponent):
    @pytest.fixture
    def components(self):
        return [
            Accumulate(),
            Accumulate(function=my_subtract),
            Accumulate(function="test.test_components.test_accumulate.my_subtract"),
        ]

    def test_accumulate_default(self):
        component = Accumulate()
        results = component.run(value=10)
        assert results == Accumulate.Output(value=10)
        assert component.state == 10

        results = component.run(value=1)
        assert results == Accumulate.Output(value=11)
        assert component.state == 11

        assert component._init_parameters == {}

    def test_accumulate_callable(self):
        component = Accumulate(function=my_subtract)

        results = component.run(value=10)
        assert results == Accumulate.Output(value=-10)
        assert component.state == -10

        results = component.run(value=1)
        assert results == Accumulate.Output(value=-11)
        assert component.state == -11

        assert component._init_parameters == {
            "function": "test.test_components.test_accumulate.my_subtract",
        }

    def test_accumulate_string(self):
        component = Accumulate(function="test.test_components.test_accumulate.my_subtract")

        results = component.run(value=10)
        assert results == Accumulate.Output(value=-10)
        assert component.state == -10

        results = component.run(value=1)
        assert results == Accumulate.Output(value=-11)
        assert component.state == -11

        assert component._init_parameters == {
            "function": "test.test_components.test_accumulate.my_subtract",
        }
