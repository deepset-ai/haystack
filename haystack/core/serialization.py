# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import inspect
from collections.abc import Callable, Iterable
from dataclasses import dataclass, is_dataclass
from typing import TYPE_CHECKING, Any, TypeVar, get_args, get_origin

from pydantic import BaseModel, TypeAdapter

from haystack.core.component.component import _hook_component_init
from haystack.core.errors import DeserializationError, SerializationError

# `allow_deserialization_module` is re-exported here to enable all serialization-specific imports
# from haystack.core.serialization.
# The redundant `as` alias marks it as an intentional re-export so ruff does not flag it (F401).
from haystack.core.serialization_security import allow_deserialization_module as allow_deserialization_module
from haystack.utils.auth import Secret
from haystack.utils.device import ComponentDevice
from haystack.utils.type_serialization import _import_class_by_name, _is_union_type

if TYPE_CHECKING:
    from haystack.core.pipeline.base import PipelineBase

T = TypeVar("T")


@dataclass(frozen=True)
class DeserializationCallbacks:
    """
    Callback functions that are invoked in specific stages of the pipeline deserialization process.

    :param component_pre_init:
        Invoked just before a component instance is
        initialized. Receives the following inputs:
        `component_name` (`str`), `component_class` (`Type`), `init_params` (`dict[str, Any]`).

        The callback is allowed to modify the `init_params`
        dictionary, which contains all the parameters that
        are passed to the component's constructor.
    """

    component_pre_init: Callable | None = None


def component_to_dict(obj: Any, name: str) -> dict[str, Any]:
    """
    Converts a component instance into a dictionary.

    If a `to_dict` method is present in the component instance, that will be used instead of the default method.

    :param obj:
        The component to be serialized.
    :param name:
        The name of the component.
    :returns:
        A dictionary representation of the component.

    :raises SerializationError:
        If the component doesn't have a `to_dict` method.
        If the values of the init parameters can't be determined.
        If a non-basic Python type is used in the serialized data.
    """
    if hasattr(obj, "to_dict"):
        data = obj.to_dict()
    else:
        init_parameters = {}
        for param_name, param in inspect.signature(obj.__init__).parameters.items():
            # Ignore `args` and `kwargs`, used by the default constructor
            if param_name in ("args", "kwargs"):
                continue
            try:
                # This only works if the Component constructor assigns the init
                # parameter to an instance variable or property with the same name
                param_value = getattr(obj, param_name)
            except AttributeError as e:
                # If the parameter doesn't have a default value, raise an error
                if param.default is param.empty:
                    raise SerializationError(
                        f"Cannot determine the value of the init parameter '{param_name}' "
                        f"for the class {obj.__class__.__name__}."
                        f"You can fix this error by assigning 'self.{param_name} = {param_name}' or adding a "
                        f"custom serialization method 'to_dict' to the class."
                    ) from e
                # In case the init parameter was not assigned, we use the default value
                param_value = param.default
            init_parameters[param_name] = param_value

        data = default_to_dict(obj, **init_parameters)

    _validate_component_to_dict_output(obj, name, data)
    return data


def _validate_component_to_dict_output(component: Any, name: str, data: dict[str, Any]) -> None:
    # Ensure that only basic Python types are used in the serde data.
    def is_allowed_type(obj: Any) -> bool:
        return isinstance(obj, (str, int, float, bool, list, dict, set, tuple, type(None)))

    def check_iterable(iterable: Iterable[Any]) -> None:
        for v in iterable:
            if not is_allowed_type(v):
                raise SerializationError(
                    f"Component '{name}' of type '{type(component).__name__}' has an unsupported value "
                    f"of type '{type(v).__name__}' in the serialized data."
                )
            if isinstance(v, (list, set, tuple)):
                check_iterable(v)
            elif isinstance(v, dict):
                check_dict(v)

    def check_dict(d: dict[str, Any]) -> None:
        if any(not isinstance(k, str) for k in d):
            raise SerializationError(
                f"Component '{name}' of type '{type(component).__name__}' has a non-string key in the serialized data."
            )

        for k, v in d.items():
            if not is_allowed_type(v):
                raise SerializationError(
                    f"Component '{name}' of type '{type(component).__name__}' has an unsupported value "
                    f"of type '{type(v).__name__}' in the serialized data under key '{k}'."
                )
            if isinstance(v, (list, set, tuple)):
                check_iterable(v)
            elif isinstance(v, dict):
                check_dict(v)

    check_dict(data)


def generate_qualified_class_name(cls: type[object]) -> str:
    """
    Generates a qualified class name for a class.

    :param cls:
        The class whose qualified name is to be generated.
    :returns:
        The qualified name of the class.
    """
    return f"{cls.__module__}.{cls.__name__}"


def component_from_dict(
    cls: type[object], data: dict[str, Any], name: str, callbacks: DeserializationCallbacks | None = None
) -> Any:
    """
    Creates a component instance from a dictionary.

    If a `from_dict` method is present in the component class, that will be used instead of the default method.

    :param cls:
        The class to be used for deserialization.
    :param data:
        The serialized data.
    :param name:
        The name of the component.
    :param callbacks:
        Callbacks to invoke during deserialization.
    :returns:
        The deserialized component.
    """

    def component_pre_init_callback(component_cls: type, init_params: dict[str, Any]) -> None:
        assert callbacks is not None
        assert callbacks.component_pre_init is not None
        callbacks.component_pre_init(name, component_cls, init_params)

    def do_from_dict() -> Any:
        if hasattr(cls, "from_dict"):
            return cls.from_dict(data)

        return default_from_dict(cls, data)

    if callbacks is None or callbacks.component_pre_init is None:
        return do_from_dict()

    with _hook_component_init(component_pre_init_callback):
        return do_from_dict()


def default_to_dict(obj: Any, **init_parameters: Any) -> dict[str, Any]:
    """
    Utility function to serialize an object to a dictionary.

    This is mostly necessary for components but can be used by any object.
    `init_parameters` are parameters passed to the object class `__init__`.
    They must be defined explicitly as they'll be used when creating a new
    instance of `obj` with `from_dict`. Omitting them might cause deserialisation
    errors or unexpected behaviours later, when calling `from_dict`.

    Objects in `init_parameters` that have a `to_dict()` method are automatically
    serialized by calling that method.

    This is the format used for saved pipeline files (`Pipeline.dump`/`Pipeline.load`). Don't merge
    it with `base_serialization._serialize_value_with_schema` — that one uses a different envelope
    for a different job (arbitrary runtime values, not Components) and changing either would break
    saved files.

    An example usage:

    ```python
    class MyClass:
        def __init__(self, my_param: int = 10) -> None:
            self.my_param = my_param

        def to_dict(self):
            return default_to_dict(self, my_param=self.my_param)


    obj = MyClass(my_param=1000)
    data = obj.to_dict()
    assert data == {
        "type": "MyClass",
        "init_parameters": {
            "my_param": 1000,
        },
    }
    ```

    :param obj:
        The object to be serialized.
    :param init_parameters:
        The parameters used to create a new instance of the class.
    :returns:
        A dictionary representation of the instance.
    """
    # Automatically serialize objects that have a to_dict method
    serialized_params = {}
    for key, value in init_parameters.items():
        if value is not None and hasattr(value, "to_dict") and callable(value.to_dict):
            serialized_params[key] = value.to_dict()
        else:
            serialized_params[key] = value

    return {"type": generate_qualified_class_name(type(obj)), "init_parameters": serialized_params}


def _is_serialized_component_device(value: Any) -> bool:
    """
    Check if a value is a serialized ComponentDevice dictionary.

    A dictionary is considered a serialized ComponentDevice if:
    - It has "type": "single" and a "device" key with a string value, or
    - It has "type": "multiple" and a "device_map" key with a dict value

    This matches the structure produced by ComponentDevice.to_dict().
    """
    if not isinstance(value, dict):
        return False

    type_value = value.get("type")
    if type_value == "single":
        return "device" in value and isinstance(value["device"], str)
    if type_value == "multiple":
        return "device_map" in value and isinstance(value["device_map"], dict)
    return False


def default_from_dict(cls: type[T], data: dict[str, Any]) -> T:
    """
    Utility function to deserialize a dictionary to an object.

    This is mostly necessary for components but can be used by any object. Reverses the
    `{"type": ..., "init_parameters": ...}` envelope produced by `default_to_dict` — see that
    function's docstring for why this envelope is not interchangeable with
    `haystack.utils.base_serialization._serialize_value_with_schema`'s.

    The function will raise a `DeserializationError` if the `type` field in `data` is
    missing or it doesn't match the type of `cls`.

    If `data` contains an `init_parameters` field it will be used as parameters to create
    a new instance of `cls`.

    Serialized Secret dictionaries in `init_parameters` are automatically detected and
    deserialized. A dictionary is considered a serialized Secret if it has a "type" key
    with value "env_var".

    Serialized ComponentDevice dictionaries in `init_parameters` are automatically detected
    and deserialized. A dictionary is considered a serialized ComponentDevice if it has a
    "type" key with value "single" or "multiple".

    Objects in `init_parameters` that are dictionaries with a "type" key containing a fully
    qualified class name are automatically detected and deserialized if the class has a
    `from_dict()` method.

    :param cls:
        The class to be used for deserialization.
    :param data:
        The serialized data.
    :returns:
        The deserialized object.

    :raises DeserializationError:
        If the `type` field in `data` is missing or it doesn't match the type of `cls`.
    """
    # Copy so that replacing serialized sub-objects (Secret/ComponentDevice/nested components) with their
    # deserialized instances below does not mutate the caller's ``data`` dict in place. Without this, a second
    # deserialization of the same dict would receive already-parsed objects instead of their serialized form.
    init_params = dict(data.get("init_parameters", {}))
    if "type" not in data:
        raise DeserializationError("Missing 'type' in serialization data")
    if data["type"] != generate_qualified_class_name(cls):
        raise DeserializationError(f"Class '{data['type']}' can't be deserialized as '{cls.__name__}'")

    valid_init_param_names = _init_parameter_names(cls)

    # Automatically detect and deserialize objects with from_dict methods
    for key, value in init_params.items():
        if isinstance(value, dict) and "type" in value:
            type_value = value.get("type")
            # Special handling for Secret (type == "env_var")
            if type_value == "env_var":
                init_params[key] = Secret.from_dict(value)
            # Special handling for ComponentDevice (type == "single" or "multiple")
            elif _is_serialized_component_device(value):
                init_params[key] = ComponentDevice.from_dict(value)
            # If type looks like a fully qualified class name, try to import it and deserialize
            elif isinstance(type_value, str) and "." in type_value:
                # Reject before importing if the parent class does not accept this parameter.
                # This blocks YAML that smuggles untrusted classes into unused parameter slots.
                if valid_init_param_names is not None and key not in valid_init_param_names:
                    known_params = (
                        f"Valid parameters are: {', '.join(repr(n) for n in sorted(valid_init_param_names))}."
                        if valid_init_param_names
                        else f"'{cls.__name__}' accepts no init parameters."
                    )
                    raise DeserializationError(
                        f"Refusing to deserialize unknown parameter '{key}' for '{cls.__name__}'. {known_params} "
                        f"Correct the parameter name or remove it from the serialized data."
                    )
                try:
                    imported_class = import_class_by_name(type_value)
                    if hasattr(imported_class, "from_dict") and callable(imported_class.from_dict):
                        init_params[key] = imported_class.from_dict(value)
                    else:
                        init_params[key] = default_from_dict(imported_class, value)
                except (ImportError, DeserializationError) as e:
                    raise type(e)(f"Failed to deserialize '{key}': {e}") from e

    return cls(**init_params)


def _init_parameter_names(cls: type[object]) -> set[str] | None:
    """
    Return the set of init parameter names accepted by `cls`.

    Returns `None` if the constructor accepts arbitrary keyword arguments (`**kwargs`) — in
    which case we cannot validate keys.
    """
    try:
        signature = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return None
    names: set[str] = set()
    for name, param in signature.parameters.items():
        if name == "self":
            continue
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            # Constructor accepts **kwargs; we cannot tell whether `key` is a real parameter.
            return None
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            continue
        names.add(name)
    return names


def import_class_by_name(fully_qualified_name: str) -> type[object]:
    """
    Utility function to import (load) a class object based on its fully qualified class name.

    This function dynamically imports a class based on its string name.
    It splits the name into module path and class name, imports the module,
    and returns the class object.

    For security, the module path is checked against the deserialization allowlist
    (see :mod:`haystack.core.serialization_security`). Modules outside the allowlist
    are rejected with a :class:`DeserializationError`.

    :param fully_qualified_name: the fully qualified class name as a string
    :returns: the class object.
    :raises ImportError: If the class cannot be imported or found.
    :raises DeserializationError: If the module is not on the deserialization allowlist.
    """
    return _import_class_by_name(fully_qualified_name)


def _coercible_classes(type_: Any) -> set[type]:
    """
    Collect the distinct classes involved in `type_` that can deserialize a plain dictionary.

    A class is coercible if it exposes a `from_dict` method (Haystack dataclasses and Components), is a Pydantic
    model, or is a standard-library dataclass. Follows `list[T]` and optionals/unions of those.

    :param type_: The type to inspect.
    :returns: The set of coercible classes `type_` involves (empty if none).
    """
    if _is_union_type(type_):
        classes: set[type] = set()
        for arm in get_args(type_):
            classes |= _coercible_classes(arm)
        return classes
    if get_origin(type_) is list:
        args = get_args(type_)
        return _coercible_classes(args[0]) if args else set()
    if isinstance(type_, type) and (hasattr(type_, "from_dict") or issubclass(type_, BaseModel) or is_dataclass(type_)):
        return {type_}
    return set()


def _deserialize_one(value: dict[str, Any], target_class: Any) -> Any:
    """
    Deserialize a single dictionary into an instance of `target_class`.

    `from_dict`-capable classes take priority so Haystack objects keep their native deserialization. Pydantic
    models and standard-library dataclasses are deserialized with Pydantic.

    :param value: The dictionary to deserialize.
    :param target_class: The coercible class resolved for the value's socket type.
    :returns: The deserialized instance.
    """
    if hasattr(target_class, "from_dict"):
        return target_class.from_dict(value)
    if issubclass(target_class, BaseModel):
        return target_class.model_validate(value)
    return TypeAdapter(target_class).validate_python(value)


def _deserialize_from_dict(value: Any, target_class: Any) -> Any:
    """
    Deserialize `value` into instances of `target_class`.

    Dictionaries are deserialized directly, lists element-wise (leaving non-dictionary items untouched). Any other
    value is returned unchanged.

    :param value: The value to deserialize.
    :param target_class: The coercible class resolved for the value's socket type.
    :returns: The deserialized value.
    """
    if isinstance(value, dict):
        return _deserialize_one(value, target_class)
    if isinstance(value, list):
        return [_deserialize_one(item, target_class) if isinstance(item, dict) else item for item in value]
    return value


def _needs_coercion(value: Any) -> bool:
    """Whether `value` carries dictionaries that could be deserialized (a dict, or a list containing one)."""
    return isinstance(value, dict) or (isinstance(value, list) and any(isinstance(item, dict) for item in value))


def _coerce_input_value(value: Any, socket_types: list[Any]) -> Any:
    if not _needs_coercion(value):
        return value
    for type_ in socket_types:
        classes = _coercible_classes(type_)
        if len(classes) > 1:
            names = ", ".join(sorted(cls.__name__ for cls in classes))
            raise DeserializationError(
                f"Cannot coerce input for socket type '{type_}': it has multiple deserializable members "
                f"({names}) and the serialized payload carries no type information to choose between them. "
                f"Provide this input as an already-deserialized object."
            )
        if classes:
            return _deserialize_from_dict(value, next(iter(classes)))
    return value


def coerce_pipeline_inputs(pipeline: "PipelineBase", data: dict[str, Any]) -> dict[str, Any]:
    """
    Deserialize serialized Haystack objects in pipeline input data, based on the pipeline's input socket types.

    For every provided value whose input socket type involves a coercible class, plain dictionaries are
    converted into instances of that class. A class is coercible if it exposes a `from_dict` method (such as
    `ChatMessage` or `Document`), is a Pydantic model, or is a standard-library dataclass. Socket types of the
    form `T`, `list[T]`, and optionals/unions of those are supported. Values that are already deserialized, or
    whose socket types involve no coercible class, are returned unchanged. A socket type that involves more than
    one distinct coercible class (such as `GeneratedAnswer | ExtractedAnswer`) is ambiguous, since the payload
    carries no type information to select one; coercing a dictionary against it raises a `DeserializationError`.

    The dictionaries are expected to be in the format produced by the class's serialization: `to_dict` for
    `from_dict`-capable classes, `model_dump` for Pydantic models, and `dataclasses.asdict` for standard-library
    dataclasses. A Pydantic model or dataclass may nest `from_dict`-capable Haystack objects: Pydantic
    (de)serializes them via their fields, so a `model_dump` payload round-trips through `model_validate` without
    going through `to_dict`/`from_dict`.

    Like `Pipeline.run`, `data` accepts the nested format (`{"component_name": {"input_name": value}}`) and
    the flat format (`{"input_name": value}`). The same format detection rules apply and the returned
    dictionary keeps the format of `data`.

    Usage example:
    ```python
    serialized_inputs = {"agent": {"messages": [message.to_dict() for message in messages]}}
    inputs = coerce_pipeline_inputs(pipeline, serialized_inputs)
    result = pipeline.run(inputs)
    ```

    :param pipeline: The pipeline whose input socket types drive the coercion.
    :param data: The pipeline input data, in nested or flat format.
    :returns: A new dictionary with the same structure as `data` and deserialized values.
    :raises DeserializationError: If a socket type involves more than one distinct coercible class and the
        corresponding value is a serialized dictionary.
    :raises Exception: Any error raised while deserializing a dictionary that does not match the expected format,
        such as a `from_dict` error or a Pydantic `ValidationError`.
    """
    # mirrors the format detection in Pipeline.run: nested if all values are dictionaries
    if all(isinstance(value, dict) for value in data.values()):
        available_inputs = pipeline.inputs(include_components_with_connected_inputs=True)
        coerced: dict[str, Any] = {}
        for component_name, component_inputs in data.items():
            sockets = available_inputs.get(component_name, {})
            coerced[component_name] = {
                input_name: _coerce_input_value(value, [sockets[input_name]["type"]] if input_name in sockets else [])
                for input_name, value in component_inputs.items()
            }
        return coerced

    # flat format: input names are matched across components like in Pipeline.run
    available_inputs = pipeline.inputs()
    return {
        input_name: _coerce_input_value(
            value, [sockets[input_name]["type"] for sockets in available_inputs.values() if input_name in sockets]
        )
        for input_name, value in data.items()
    }
