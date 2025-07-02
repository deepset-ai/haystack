# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import requests

from haystack.components.generators.utils import print_streaming_chunk
from haystack.core.errors import DeserializationError, SerializationError
from haystack.testing.callable_serialization.random_callable import callable_to_deserialize
from haystack.utils import deserialize_callable, serialize_callable


def some_random_callable_for_testing(some_ignored_arg: str):
    pass


class TestClass:
    @classmethod
    def class_method(cls):
        pass

    @staticmethod
    def static_method():
        pass

    def my_method(self):
        pass


def test_callable_serialization():
    result = serialize_callable(some_random_callable_for_testing)
    assert result == "test_callable_serialization.some_random_callable_for_testing"


def test_callable_serialization_non_local():
    # check our callable serialization
    result = serialize_callable(print_streaming_chunk)
    assert result == "haystack.components.generators.utils.print_streaming_chunk"

    # check serialization of another library's callable
    result = serialize_callable(requests.api.get)
    assert result == "requests.api.get"


def test_fully_qualified_import_deserialization():
    func = deserialize_callable("haystack.testing.callable_serialization.random_callable.callable_to_deserialize")

    assert func is callable_to_deserialize
    assert func("Hello") == "Hello, world!"


def test_callable_serialization_instance_methods_fail():
    with pytest.raises(SerializationError):
        serialize_callable(TestClass.my_method)

    instance = TestClass()
    with pytest.raises(SerializationError):
        serialize_callable(instance.my_method)


def test_lambda_serialization_fail():
    with pytest.raises(SerializationError):
        serialize_callable(lambda x: x)


def test_nested_function_serialization_fail():
    def my_fun():
        pass

    with pytest.raises(SerializationError):
        serialize_callable(my_fun)


def test_callable_deserialization():
    result = serialize_callable(some_random_callable_for_testing)
    fn = deserialize_callable(result)
    assert fn is some_random_callable_for_testing


def test_callable_deserialization_non_local():
    result = serialize_callable(requests.api.get)
    fn = deserialize_callable(result)
    assert fn is requests.api.get


def test_classmethod_serialization_deserialization():
    result = serialize_callable(TestClass.class_method)
    fn = deserialize_callable(result)
    assert fn == TestClass.class_method


def test_staticmethod_serialization_deserialization():
    result = serialize_callable(TestClass.static_method)
    fn = deserialize_callable(result)
    assert fn == TestClass.static_method


def test_callable_deserialization_errors():
    # module does not exist
    with pytest.raises(DeserializationError):
        deserialize_callable("nonexistent_module.function")

    # function does not exist
    with pytest.raises(DeserializationError):
        deserialize_callable("os.nonexistent_function")

    # attribute is not callable
    with pytest.raises(DeserializationError):
        deserialize_callable("os.name")
