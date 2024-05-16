# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import copy
import sys
import typing
from typing import List, Dict

import pytest

from haystack.dataclasses import ChatMessage
from haystack.components.routers.conditional_router import serialize_type, deserialize_type


def test_output_type_serialization():
    assert serialize_type(str) == "str"
    assert serialize_type(List[int]) == "typing.List[int]"
    assert serialize_type(List[Dict[str, int]]) == "typing.List[typing.Dict[str, int]]"
    assert serialize_type(ChatMessage) == "haystack.dataclasses.chat_message.ChatMessage"
    assert serialize_type(typing.List[Dict[str, int]]) == "typing.List[typing.Dict[str, int]]"
    assert serialize_type(List[ChatMessage]) == "typing.List[haystack.dataclasses.chat_message.ChatMessage]"
    assert (
        serialize_type(typing.Dict[int, ChatMessage])
        == "typing.Dict[int, haystack.dataclasses.chat_message.ChatMessage]"
    )
    assert serialize_type(int) == "int"
    assert serialize_type(ChatMessage.from_user("ciao")) == "haystack.dataclasses.chat_message.ChatMessage"


@pytest.mark.skipif(sys.version_info < (3, 9), reason="PEP 585 types are only available in Python 3.9+")
def test_output_type_serialization_pep585():
    # Only Python 3.9+ supports PEP 585 types and can serialize them
    # PEP 585 types
    assert serialize_type(list[int]) == "list[int]"
    assert serialize_type(list[list[int]]) == "list[list[int]]"

    # more nested types
    assert serialize_type(list[list[list[int]]]) == "list[list[list[int]]]"
    assert serialize_type(dict[str, int]) == "dict[str, int]"


def test_output_type_deserialization():
    assert deserialize_type("str") == str
    assert deserialize_type("typing.List[int]") == typing.List[int]
    assert deserialize_type("typing.List[typing.Dict[str, int]]") == typing.List[Dict[str, int]]
    assert deserialize_type("typing.Dict[str, int]") == Dict[str, int]
    assert deserialize_type("typing.Dict[str, typing.List[int]]") == Dict[str, List[int]]
    assert deserialize_type("typing.List[typing.Dict[str, typing.List[int]]]") == List[Dict[str, List[int]]]
    assert deserialize_type("typing.List[haystack.dataclasses.chat_message.ChatMessage]") == typing.List[ChatMessage]
    assert (
        deserialize_type("typing.Dict[int, haystack.dataclasses.chat_message.ChatMessage]")
        == typing.Dict[int, ChatMessage]
    )
    assert deserialize_type("haystack.dataclasses.chat_message.ChatMessage") == ChatMessage
    assert deserialize_type("int") == int


def test_output_type_deserialization_pep585():
    is_pep585 = sys.version_info >= (3, 9)

    # Although only Python 3.9+ supports PEP 585 types, we can still deserialize them in older Python versions
    # as their typing equivalents
    assert deserialize_type("list[int]") == list[int] if is_pep585 else List[int]
    assert deserialize_type("dict[str, int]") == dict[str, int] if is_pep585 else Dict[str, int]
    # more nested types
    assert deserialize_type("list[list[int]]") == list[list[int]] if is_pep585 else List[List[int]]
    assert deserialize_type("list[list[list[int]]]") == list[list[list[int]]] if is_pep585 else List[List[List[int]]]
