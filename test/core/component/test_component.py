# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import logging
from functools import partial
from typing import Any

import pytest

from haystack.core.component import Component, InputSocket, OutputSocket, component
from haystack.core.component.component import _hook_component_init
from haystack.core.component.types import Variadic
from haystack.core.errors import ComponentError
from haystack.core.pipeline import Pipeline


def test_correct_declaration():
    @component
    class MockComponent:
        def to_dict(self):
            return {}

        @classmethod
        def from_dict(cls, data):
            return cls()

        @component.output_types(output_value=int)
        def run(self, input_value: int):
            return {"output_value": input_value}

    # Verifies also instantiation works with no issues
    assert MockComponent()
    assert component.registry["test_component.MockComponent"] == MockComponent
    assert isinstance(MockComponent(), Component)
    assert MockComponent().__haystack_supports_async__ is False


def test_correct_declaration_with_async():
    @component
    class MockComponent:
        def to_dict(self):
            return {}

        @classmethod
        def from_dict(cls, data):
            return cls()

        @component.output_types(output_value=int)
        def run(self, input_value: int):
            return {"output_value": input_value}

        @component.output_types(output_value=int)
        async def async_run(self, input_value: int):
            return {"output_value": input_value}

    # Verifies also instantiation works with no issues
    assert MockComponent()
    assert component.registry["test_component.MockComponent"] == MockComponent
    assert isinstance(MockComponent(), Component)
    assert MockComponent().__haystack_supports_async__ is True


def test_correct_declaration_with_additional_readonly_property():
    @component
    class MockComponent:
        @property
        def store(self):
            return "test_store"

        def to_dict(self):
            return {}

        @classmethod
        def from_dict(cls, data):
            return cls()

        @component.output_types(output_value=int)
        def run(self, input_value: int):
            return {"output_value": input_value}

    # Verifies that instantiation works with no issues
    assert MockComponent()
    assert component.registry["test_component.MockComponent"] == MockComponent
    assert MockComponent().store == "test_store"


def test_correct_declaration_with_additional_writable_property():
    @component
    class MockComponent:
        @property
        def store(self):
            return "test_store"

        @store.setter
        def store(self, value):
            self._store = value

        def to_dict(self):
            return {}

        @classmethod
        def from_dict(cls, data):
            return cls()

        @component.output_types(output_value=int)
        def run(self, input_value: int):
            return {"output_value": input_value}

    # Verifies that instantiation works with no issues
    assert component.registry["test_component.MockComponent"] == MockComponent
    comp = MockComponent()
    comp.store = "test_store"
    assert comp.store == "test_store"


def test_missing_run():
    with pytest.raises(ComponentError, match=r"must have a 'run\(\)' method"):

        @component
        class MockComponent:
            def another_method(self, input_value: int):
                return {"output_value": input_value}


def test_async_run_not_async():
    @component
    class MockComponent:
        @component.output_types(value=int)
        def run(self, value: int):
            return {"value": 1}

        @component.output_types(value=int)
        def async_run(self, value: int):
            return {"value": 1}

    with pytest.raises(ComponentError):
        comp = MockComponent()


def test_async_run_not_coroutine():
    @component
    class MockComponent:
        @component.output_types(value=int)
        def run(self, value: int):
            return {"value": 1}

        @component.output_types(value=int)
        async def async_run(self, value: int):
            yield {"value": 1}

    with pytest.raises(ComponentError):
        comp = MockComponent()


def test_parameters_mismatch_run_and_async_run():
    @component
    class MockComponent:
        @component.output_types(value=int)
        def run(self, value: int):
            return {"value": 1}

        async def async_run(self, value: str):
            yield {"value": "1"}

    with pytest.raises(ComponentError):
        comp = MockComponent()


def test_set_input_types():
    @component
    class MockComponent:
        def __init__(self):
            component.set_input_types(self, value=Any)

        def to_dict(self):
            return {}

        @classmethod
        def from_dict(cls, data):
            return cls()

        @component.output_types(value=int)
        def run(self, **kwargs):
            return {"value": 1}

    comp = MockComponent()
    assert comp.__haystack_input__._sockets_dict == {"value": InputSocket("value", Any)}
    assert comp.run() == {"value": 1}


def test_set_output_types():
    @component
    class MockComponent:
        def __init__(self):
            component.set_output_types(self, value=int)

        def to_dict(self):
            return {}

        @classmethod
        def from_dict(cls, data):
            return cls()

        def run(self, value: int):
            return {"value": 1}

    comp = MockComponent()
    assert comp.__haystack_output__._sockets_dict == {"value": OutputSocket("value", int)}


def test_output_types_decorator_with_compatible_type():
    @component
    class MockComponent:
        @component.output_types(value=int)
        def run(self, value: int):
            return {"value": 1}

        def to_dict(self):
            return {}

        @classmethod
        def from_dict(cls, data):
            return cls()

    comp = MockComponent()
    assert comp.__haystack_output__._sockets_dict == {"value": OutputSocket("value", int)}


def test_output_types_decorator_wrong_method():
    with pytest.raises(ComponentError):

        @component
        class MockComponent:
            def run(self, value: int):
                return {"value": 1}

            @component.output_types(value=int)
            def to_dict(self):
                return {}

            @classmethod
            def from_dict(cls, data):
                return cls()


def test_output_types_decorator_mismatch_run_async_run():
    @component
    class MockComponent:
        @component.output_types(value=int)
        def run(self, value: int):
            return {"value": 1}

        @component.output_types(value=str)
        async def async_run(self, value: int):
            return {"value": "1"}

    with pytest.raises(ComponentError):
        comp = MockComponent()


def test_output_types_decorator_missing_async_run():
    @component
    class MockComponent:
        @component.output_types(value=int)
        def run(self, value: int):
            return {"value": 1}

        async def async_run(self, value: int):
            return {"value": "1"}

    with pytest.raises(ComponentError):
        comp = MockComponent()


def test_component_decorator_set_it_as_component():
    @component
    class MockComponent:
        @component.output_types(value=int)
        def run(self, value: int):
            return {"value": 1}

        def to_dict(self):
            return {}

        @classmethod
        def from_dict(cls, data):
            return cls()

    comp = MockComponent()
    assert isinstance(comp, Component)


def test_input_has_default_value():
    @component
    class MockComponent:
        @component.output_types(value=int)
        def run(self, value: int = 42):
            return {"value": value}

    comp = MockComponent()
    assert comp.__haystack_input__._sockets_dict["value"].default_value == 42
    assert not comp.__haystack_input__._sockets_dict["value"].is_mandatory


def test_keyword_only_args():
    @component
    class MockComponent:
        def __init__(self):
            component.set_output_types(self, value=int)

        def run(self, *, arg: int):
            return {"value": arg}

    comp = MockComponent()
    component_inputs = {name: {"type": socket.type} for name, socket in comp.__haystack_input__._sockets_dict.items()}
    assert component_inputs == {"arg": {"type": int}}


def test_repr():
    @component
    class MockComponent:
        def __init__(self):
            component.set_output_types(self, value=int)

        def run(self, value: int):
            return {"value": value}

    comp = MockComponent()
    assert repr(comp) == f"{object.__repr__(comp)}\nInputs:\n  - value: int\nOutputs:\n  - value: int"


def test_repr_added_to_pipeline():
    @component
    class MockComponent:
        def __init__(self):
            component.set_output_types(self, value=int)

        def run(self, value: int):
            return {"value": value}

    pipe = Pipeline()
    comp = MockComponent()
    pipe.add_component("my_component", comp)
    assert repr(comp) == f"{object.__repr__(comp)}\nmy_component\nInputs:\n  - value: int\nOutputs:\n  - value: int"


def test_is_greedy_default_with_variadic_input():
    @component
    class MockComponent:
        @component.output_types(value=int)
        def run(self, value: Variadic[int]):
            return {"value": value}

    assert not MockComponent.__haystack_is_greedy__
    assert not MockComponent().__haystack_is_greedy__


def test_is_greedy_default_without_variadic_input():
    @component
    class MockComponent:
        @component.output_types(value=int)
        def run(self, value: int):
            return {"value": value}

    assert not MockComponent.__haystack_is_greedy__
    assert not MockComponent().__haystack_is_greedy__


def test_is_greedy_flag_with_variadic_input():
    @component(is_greedy=True)
    class MockComponent:
        @component.output_types(value=int)
        def run(self, value: Variadic[int]):
            return {"value": value}

    assert MockComponent.__haystack_is_greedy__
    assert MockComponent().__haystack_is_greedy__


def test_is_greedy_flag_without_variadic_input(caplog):
    caplog.set_level(logging.WARNING)

    @component(is_greedy=True)
    class MockComponent:
        @component.output_types(value=int)
        def run(self, value: int):
            return {"value": value}

    assert MockComponent.__haystack_is_greedy__
    assert caplog.text == ""
    assert MockComponent().__haystack_is_greedy__
    assert (
        "Component 'MockComponent' has no variadic input, but it's marked as greedy."
        " This is not supported and can lead to unexpected behavior.\n" in caplog.text
    )


def test_pre_init_hooking():
    @component
    class MockComponent:
        def __init__(self, pos_arg1, pos_arg2, pos_arg3=None, *, kwarg1=1, kwarg2="string"):
            self.pos_arg1 = pos_arg1
            self.pos_arg2 = pos_arg2
            self.pos_arg3 = pos_arg3
            self.kwarg1 = kwarg1
            self.kwarg2 = kwarg2

        @component.output_types(output_value=int)
        def run(self, input_value: int):
            return {"output_value": input_value}

    def pre_init_hook(component_class, init_params, expected_params):
        assert component_class == MockComponent
        assert init_params == expected_params

    def pre_init_hook_modify(component_class, init_params, expected_params):
        assert component_class == MockComponent
        assert init_params == expected_params

        init_params["pos_arg1"] = 2
        init_params["pos_arg2"] = 0
        init_params["pos_arg3"] = "modified"
        init_params["kwarg2"] = "modified string"

    with _hook_component_init(partial(pre_init_hook, expected_params={"pos_arg1": 1, "pos_arg2": 2, "kwarg1": None})):
        _ = MockComponent(1, 2, kwarg1=None)

    with _hook_component_init(partial(pre_init_hook, expected_params={"pos_arg1": 1, "pos_arg2": 2, "pos_arg3": 0.01})):
        _ = MockComponent(pos_arg1=1, pos_arg2=2, pos_arg3=0.01)

    with _hook_component_init(
        partial(pre_init_hook_modify, expected_params={"pos_arg1": 0, "pos_arg2": 1, "pos_arg3": 0.01, "kwarg1": 0})
    ):
        c = MockComponent(0, 1, pos_arg3=0.01, kwarg1=0)

        assert c.pos_arg1 == 2
        assert c.pos_arg2 == 0
        assert c.pos_arg3 == "modified"
        assert c.kwarg1 == 0
        assert c.kwarg2 == "modified string"


def test_pre_init_hooking_variadic_positional_args():
    @component
    class MockComponent:
        def __init__(self, *args, kwarg1=1, kwarg2="string"):
            self.args = args
            self.kwarg1 = kwarg1
            self.kwarg2 = kwarg2

        @component.output_types(output_value=int)
        def run(self, input_value: int):
            return {"output_value": input_value}

    def pre_init_hook(component_class, init_params, expected_params):
        assert component_class == MockComponent
        assert init_params == expected_params

    c = MockComponent(1, 2, 3, kwarg1=None)
    assert c.args == (1, 2, 3)
    assert c.kwarg1 is None
    assert c.kwarg2 == "string"

    with pytest.raises(ComponentError), _hook_component_init(
        partial(pre_init_hook, expected_params={"args": (1, 2), "kwarg1": None})
    ):
        _ = MockComponent(1, 2, kwarg1=None)


def test_pre_init_hooking_variadic_kwargs():
    @component
    class MockComponent:
        def __init__(self, pos_arg1, pos_arg2=None, **kwargs):
            self.pos_arg1 = pos_arg1
            self.pos_arg2 = pos_arg2
            self.kwargs = kwargs

        @component.output_types(output_value=int)
        def run(self, input_value: int):
            return {"output_value": input_value}

    def pre_init_hook(component_class, init_params, expected_params):
        assert component_class == MockComponent
        assert init_params == expected_params

    with _hook_component_init(
        partial(pre_init_hook, expected_params={"pos_arg1": 1, "kwarg1": None, "kwarg2": 10, "kwarg3": "string"})
    ):
        c = MockComponent(1, kwarg1=None, kwarg2=10, kwarg3="string")
        assert c.pos_arg1 == 1
        assert c.pos_arg2 is None
        assert c.kwargs == {"kwarg1": None, "kwarg2": 10, "kwarg3": "string"}

    def pre_init_hook_modify(component_class, init_params, expected_params):
        assert component_class == MockComponent
        assert init_params == expected_params

        init_params["pos_arg1"] = 2
        init_params["pos_arg2"] = 0
        init_params["some_kwarg"] = "modified string"

    with _hook_component_init(
        partial(
            pre_init_hook_modify,
            expected_params={"pos_arg1": 0, "pos_arg2": 1, "kwarg1": 999, "some_kwarg": "some_value"},
        )
    ):
        c = MockComponent(0, 1, kwarg1=999, some_kwarg="some_value")

        assert c.pos_arg1 == 2
        assert c.pos_arg2 == 0
        assert c.kwargs == {"kwarg1": 999, "some_kwarg": "modified string"}
