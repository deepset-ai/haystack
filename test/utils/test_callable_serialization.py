# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import requests

from haystack.components.generators.utils import print_streaming_chunk
from haystack.utils import serialize_callable, deserialize_callable


def some_random_callable_for_testing(some_ignored_arg: str):
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


def test_callable_deserialization():
    result = serialize_callable(some_random_callable_for_testing)
    fn = deserialize_callable(result)
    assert fn is some_random_callable_for_testing


def test_callable_deserialization_non_local():
    result = serialize_callable(requests.api.get)
    fn = deserialize_callable(result)
    assert fn is requests.api.get
