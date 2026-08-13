# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import builtins
import sys
import typing
from collections import deque
from types import UnionType
from typing import Annotated, Any, Callable, Deque, Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Union

import pytest

from haystack.core.errors import DeserializationError
from haystack.core.serialization_security import _DENIED_BUILTIN_NAMES
from haystack.dataclasses import Answer, ByteStream, ChatMessage, Document
from haystack.utils.type_serialization import (
    _build_pep604_union_type,
    _is_union_type,
    _parse_pep604_union_args,
    deserialize_type,
    serialize_type,
)

TYPING_AND_TYPE_TESTS = [
    # dict
    pytest.param("dict", dict),
    pytest.param("dict[str, int]", dict[str, int]),
    pytest.param("dict[int, str]", dict[int, str]),
    pytest.param("dict[dict, dict]", dict[dict, dict]),
    pytest.param("dict[float, float]", dict[float, float]),
    pytest.param("dict[bool, bool]", dict[bool, bool]),
    # typing Dict
    pytest.param("typing.Dict", Dict),
    pytest.param("typing.Dict[str, int]", Dict[str, int]),
    pytest.param("typing.Dict[int, str]", Dict[int, str]),
    pytest.param("typing.Dict[dict, dict]", Dict[dict, dict]),
    pytest.param("typing.Dict[float, float]", Dict[float, float]),
    pytest.param("typing.Dict[bool, bool]", Dict[bool, bool]),
    # list
    pytest.param("list", list),
    pytest.param("list[int]", list[int]),
    pytest.param("list[str]", list[str]),
    pytest.param("list[dict]", list[dict]),
    pytest.param("list[float]", list[float]),
    pytest.param("list[bool]", list[bool]),
    # typing List
    pytest.param("typing.List", List),
    pytest.param("typing.List[int]", List[int]),
    pytest.param("typing.List[str]", List[str]),
    pytest.param("typing.List[dict]", List[dict]),
    pytest.param("typing.List[float]", List[float]),
    pytest.param("typing.List[bool]", List[bool]),
    # PEP 604 X | None
    pytest.param("str | None", str | None),
    pytest.param("int | None", int | None),
    pytest.param("dict | None", dict | None),
    pytest.param("float | None", float | None),
    pytest.param("bool | None", bool | None),
    pytest.param("list[str] | None", list[str] | None),
    pytest.param("dict[str, int] | None", dict[str, int] | None),
    # set
    pytest.param("set", set),
    pytest.param("set[int]", set[int]),
    pytest.param("set[str]", set[str]),
    pytest.param("set[dict]", set[dict]),
    pytest.param("set[float]", set[float]),
    pytest.param("set[bool]", set[bool]),
    # typing Set
    pytest.param("typing.Set", Set),
    pytest.param("typing.Set[int]", Set[int]),
    pytest.param("typing.Set[str]", Set[str]),
    pytest.param("typing.Set[dict]", Set[dict]),
    pytest.param("typing.Set[float]", Set[float]),
    pytest.param("typing.Set[bool]", Set[bool]),
    # tuple
    pytest.param("tuple", tuple),
    pytest.param("tuple[int]", tuple[int]),
    pytest.param("tuple[str]", tuple[str]),
    pytest.param("tuple[dict]", tuple[dict]),
    pytest.param("tuple[float]", tuple[float]),
    pytest.param("tuple[bool]", tuple[bool]),
    # variadic tuple (the `...` is the Ellipsis singleton, not a type)
    pytest.param("tuple[int, ...]", tuple[int, ...]),
    pytest.param("tuple[str, ...]", tuple[str, ...]),
    pytest.param("tuple[dict[str, int], ...]", tuple[dict[str, int], ...]),
    # typing Tuple
    pytest.param("typing.Tuple", Tuple),
    pytest.param("typing.Tuple[int]", Tuple[int]),
    pytest.param("typing.Tuple[str]", Tuple[str]),
    pytest.param("typing.Tuple[dict]", Tuple[dict]),
    pytest.param("typing.Tuple[float]", Tuple[float]),
    pytest.param("typing.Tuple[bool]", Tuple[bool]),
    pytest.param("typing.Tuple[int, ...]", Tuple[int, ...]),
    # callable (the `...` is the Ellipsis singleton, not a type)
    pytest.param("typing.Callable[..., int]", Callable[..., int]),
    pytest.param("typing.Callable[..., str]", Callable[..., str]),
    # PEP 604 X | Y
    pytest.param("str | int", str | int),
    pytest.param("int | float", int | float),
    pytest.param("dict | str", dict | str),
    pytest.param("float | bool", float | bool),
    pytest.param("str | int | float", str | int | float),
    pytest.param("list[str] | list[int]", list[str] | list[int]),
    pytest.param("dict[str, int] | list[str]", dict[str, int] | list[str]),
    # other
    pytest.param("frozenset", frozenset),
    pytest.param("frozenset[int]", frozenset[int]),
    pytest.param("collections.deque", deque),
    pytest.param("collections.deque[str]", deque[str]),
    # typing Other
    pytest.param("typing.Any", Any),
    pytest.param("typing.FrozenSet", FrozenSet),
    pytest.param("typing.FrozenSet[int]", FrozenSet[int]),
    pytest.param("typing.Deque", Deque),
    pytest.param("typing.Deque[str]", Deque[str]),
]


@pytest.mark.parametrize("output_str, input_type", TYPING_AND_TYPE_TESTS)
def test_output_type_serialization_typing_and_type(output_str, input_type):
    assert serialize_type(input_type) == output_str


@pytest.mark.parametrize("input_str, expected_output", TYPING_AND_TYPE_TESTS)
def test_output_type_deserialization_typing_and_type(input_str, expected_output):
    assert deserialize_type(input_str) == expected_output


def test_output_type_deserialization_typing_no_module():
    assert deserialize_type("List[int]") == List[int]
    assert deserialize_type("Dict[str, int]") == Dict[str, int]
    assert deserialize_type("Set[int]") == Set[int]
    assert deserialize_type("Tuple[int]") == Tuple[int]
    assert deserialize_type("FrozenSet[int]") == FrozenSet[int]
    assert deserialize_type("Deque[str]") == Deque[str]
    assert deserialize_type("Optional[int]") == Optional[int]
    assert deserialize_type("Union[str, int]") == Union[str, int]


def test_output_type_serialization():
    assert serialize_type(str) == "str"
    assert serialize_type(int) == "int"
    assert serialize_type(dict) == "dict"
    assert serialize_type(float) == "float"
    assert serialize_type(bool) == "bool"
    assert serialize_type(None) == "None"


def test_output_type_serialization_string():
    assert serialize_type("str") == "str"
    assert serialize_type("builtins.str") == "builtins.str"


def test_output_type_deserialization():
    assert deserialize_type("str") == str
    assert deserialize_type("int") == int
    assert deserialize_type("dict") == dict
    assert deserialize_type("float") == float
    assert deserialize_type("bool") == bool
    assert deserialize_type("None") is None
    assert deserialize_type("NoneType") == type(None)


def test_output_builtin_type_deserialization():
    assert deserialize_type("builtins.str") == str
    assert deserialize_type("builtins.int") == int
    assert deserialize_type("builtins.dict") == dict
    assert deserialize_type("builtins.float") == float
    assert deserialize_type("builtins.bool") == bool


# `type` is excluded: it is a valid type in this path (covered by test_builtin_types_round_trip),
# even though it is denied as a *callable*. Every other denied builtin is a function, not a type.
@pytest.mark.parametrize("name", sorted(_DENIED_BUILTIN_NAMES - {"type"}))
def test_dangerous_builtins_rejected(name):
    # `builtins` is on the allowlist, but a type annotation must resolve to an actual type. Builtin
    # functions are rejected both with the `builtins.` prefix and via the bare-name fallback (which
    # skips the allowlist). The dunder-named ones (`__import__`, `__build_class__`) are refused even
    # earlier by the object-internals traversal guard when a `builtins.` prefix is used.
    with pytest.raises(DeserializationError, match="not a type|internal attribute"):
        deserialize_type(f"builtins.{name}")
    with pytest.raises(DeserializationError, match="not a type"):
        deserialize_type(name)


@pytest.mark.parametrize("name", ["memoryview", "type", "bytearray", "frozenset"])
def test_builtin_types_round_trip(name):
    # Builtin *types* must still resolve as annotations — the type gate keys on `isinstance(type)`,
    # not on whether the name is also callable, so e.g. `memoryview` and `type` are allowed.
    expected = getattr(builtins, name)
    assert deserialize_type(name) is expected
    assert deserialize_type(f"builtins.{name}") is expected


def test_output_type_serialization_nested():
    # typing
    assert serialize_type(List[Dict[str, int]]) == "typing.List[typing.Dict[str, int]]"
    assert serialize_type(typing.List[Dict[str, int]]) == "typing.List[typing.Dict[str, int]]"
    # builtins

    assert serialize_type(list[dict[str, int]]) == "list[dict[str, int]]"
    assert serialize_type(list[list[int]]) == "list[list[int]]"
    assert serialize_type(list[list[list[int]]]) == "list[list[list[int]]]"
    # PEP 604
    assert serialize_type(list[str | int]) == "list[str | int]"
    assert serialize_type(list[str | None]) == "list[str | None]"
    assert serialize_type(dict[str, int | None]) == "dict[str, int | None]"
    assert serialize_type(list[dict[str, int] | None]) == "list[dict[str, int] | None]"


def test_output_type_deserialization_nested():
    # typing
    assert deserialize_type("typing.List[typing.Union[str, int]]") == List[Union[str, int]]
    assert deserialize_type("typing.List[typing.Optional[str]]") == List[Optional[str]]
    assert deserialize_type("typing.List[typing.Dict[str, typing.List[int]]]") == List[Dict[str, List[int]]]
    assert deserialize_type("typing.List[typing.Dict[str, int]]") == typing.List[Dict[str, int]]
    # builtins
    assert deserialize_type("list[typing.Union[str, int]]") == list[Union[str, int]]
    assert deserialize_type("list[typing.Optional[str]]") == list[Optional[str]]
    assert deserialize_type("list[dict[str, list[int]]]") == list[dict[str, list[int]]]
    assert deserialize_type("list[dict[str, int]]") == list[dict[str, int]]
    assert deserialize_type("list[list[int]]") == list[list[int]]
    assert deserialize_type("list[list[list[int]]]") == list[list[list[int]]]
    # PEP 604
    assert deserialize_type("list[str | int]") == list[Union[str, int]]
    assert deserialize_type("list[str | None]") == list[Union[str, None]]
    assert deserialize_type("dict[str, int | None]") == dict[str, Union[int, None]]
    assert deserialize_type("list[dict[str, int] | None]") == list[Union[dict[str, int], None]]


def test_output_type_serialization_typing_generic_with_nonetype():
    # NoneType used as a regular argument of a typing generic (not the implicit None of Optional)
    # must be kept, otherwise the serialized type is malformed (e.g. "typing.Dict[str]") or loses information.
    assert serialize_type(Dict[str, type(None)]) == "typing.Dict[str, None]"  # type: ignore[misc]
    assert serialize_type(Dict[type(None), str]) == "typing.Dict[None, str]"  # type: ignore[misc]
    assert serialize_type(Tuple[int, type(None)]) == "typing.Tuple[int, None]"
    assert serialize_type(List[type(None)]) == "typing.List[None]"  # type: ignore[misc]
    # A Union with more than two members that includes None must keep None as well.
    assert serialize_type(Union[str, int, None]) == "typing.Union[str, int, None]"
    # Optional must still be serialized without a redundant trailing None.
    assert serialize_type(Optional[str]) == "typing.Optional[str]"


def test_output_type_round_trip_typing_generic_with_nonetype():
    for type_ in [
        Dict[str, type(None)],  # type: ignore[misc]
        Dict[type(None), str],  # type: ignore[misc]
        Tuple[int, type(None)],
        List[type(None)],  # type: ignore[misc]
        Union[str, int, None],
        Optional[str],
    ]:
        assert deserialize_type(serialize_type(type_)) == type_


def test_output_type_deserialization_legacy_ellipsis_literal():
    # Types serialized by older versions emitted the literal "Ellipsis"; make sure they still load.
    assert deserialize_type("tuple[int, Ellipsis]") == tuple[int, ...]
    assert deserialize_type("typing.Callable[Ellipsis, int]") == Callable[..., int]


def test_output_type_serialization_callable_with_parameter_list():
    # `Callable[[X, Y], R]` returns its parameter list from `typing.get_args` as a Python list ([X, Y]),
    # not as a type. It must be serialized as "[X, Y]" so the parameters survive the round-trip.
    assert serialize_type(Callable[[int, str], bool]) == "typing.Callable[[int, str], bool]"
    assert serialize_type(Callable[[], int]) == "typing.Callable[[], int]"
    assert serialize_type(Callable[[int], List[str]]) == "typing.Callable[[int], typing.List[str]]"
    assert serialize_type(Callable[[Union[str, int]], bool]) == "typing.Callable[[typing.Union[str, int]], bool]"


def test_output_type_deserialization_callable_with_parameter_list():
    assert deserialize_type("typing.Callable[[int, str], bool]") == Callable[[int, str], bool]
    assert deserialize_type("typing.Callable[[], int]") == Callable[[], int]
    assert deserialize_type("typing.Callable[[int], typing.List[str]]") == Callable[[int], List[str]]
    assert deserialize_type("typing.Callable[[typing.Union[str, int]], bool]") == Callable[[Union[str, int]], bool]


def test_output_type_round_trip_callable_with_parameter_list():
    for type_ in [
        Callable[[int, str], bool],
        Callable[[], int],
        Callable[[int], List[str]],
        Callable[[Dict[str, int], str], List[bool]],
        Callable[[Union[str, int]], bool],
        List[Callable[[int], str]],
        Dict[str, Callable[[int, str], bool]],
        Optional[Callable[[int], str]],
        Callable[[Callable[[int], str]], bool],
        # The Ellipsis form (no parameter list) must still round-trip unaffected by the parameter-list
        # handling above: `_serialize_type_arg`/`_deserialize_type_arg` only special-case a `list` argument,
        # so `...` still falls through to the existing Ellipsis handling in serialize_type/deserialize_type.
        Callable[..., int],
        Callable[[int], Callable[..., str]],
    ]:
        assert deserialize_type(serialize_type(type_)) == type_


def test_output_type_serialization_literal():
    assert serialize_type(Literal["yes", "no"]) == "typing.Literal['yes', 'no']"
    assert serialize_type(Literal[1, 2, 3]) == "typing.Literal[1, 2, 3]"
    assert serialize_type(Literal[True, False]) == "typing.Literal[True, False]"
    # A value that happens to look like a type name must stay a string, not be rendered as that type.
    assert serialize_type(Literal["int", "str"]) == "typing.Literal['int', 'str']"


def test_output_type_deserialization_literal():
    assert deserialize_type("typing.Literal['yes', 'no']") == Literal["yes", "no"]
    assert deserialize_type("typing.Literal[1, 2, 3]") == Literal[1, 2, 3]
    assert deserialize_type("typing.Literal[True, False]") == Literal[True, False]
    assert deserialize_type("typing.Literal['int', 'str']") == Literal["int", "str"]


def test_output_type_round_trip_literal():
    for type_ in [
        Literal["yes", "no"],
        Literal["int", "str"],  # values that look like type names must round-trip as strings, not types
        Literal[1, 2, 3],
        Literal[True, False],
        Literal["None"],  # the string "None", not the NoneType singleton
        Literal["a, b", "c"],  # a comma inside a value must not split the arguments
        Literal["x"],
        Literal[b"bytes"],
        Optional[Literal["a", "b"]],
        Union[Literal["a"], int],
    ]:
        assert deserialize_type(serialize_type(type_)) == type_


def test_output_type_serialization_haystack_dataclasses():
    # typing
    # Answer
    assert serialize_type(Answer) == "haystack.dataclasses.answer.Answer"
    assert serialize_type(List[Answer]) == "typing.List[haystack.dataclasses.answer.Answer]"
    assert serialize_type(typing.Dict[int, Answer]) == "typing.Dict[int, haystack.dataclasses.answer.Answer]"
    # Bytestream
    assert serialize_type(ByteStream) == "haystack.dataclasses.byte_stream.ByteStream"
    assert serialize_type(List[ByteStream]) == "typing.List[haystack.dataclasses.byte_stream.ByteStream]"
    assert (
        serialize_type(typing.Dict[int, ByteStream]) == "typing.Dict[int, haystack.dataclasses.byte_stream.ByteStream]"
    )
    # Chat Message
    assert serialize_type(ChatMessage) == "haystack.dataclasses.chat_message.ChatMessage"
    assert serialize_type(List[ChatMessage]) == "typing.List[haystack.dataclasses.chat_message.ChatMessage]"
    assert (
        serialize_type(typing.Dict[int, ChatMessage])
        == "typing.Dict[int, haystack.dataclasses.chat_message.ChatMessage]"
    )
    # Document
    assert serialize_type(Document) == "haystack.dataclasses.document.Document"
    assert serialize_type(List[Document]) == "typing.List[haystack.dataclasses.document.Document]"
    assert serialize_type(typing.Dict[int, Document]) == "typing.Dict[int, haystack.dataclasses.document.Document]"
    # builtins
    # Answer
    assert serialize_type(list[Answer]) == "list[haystack.dataclasses.answer.Answer]"
    assert serialize_type(dict[int, Answer]) == "dict[int, haystack.dataclasses.answer.Answer]"
    # Bytestream
    assert serialize_type(list[ByteStream]) == "list[haystack.dataclasses.byte_stream.ByteStream]"
    assert serialize_type(dict[int, ByteStream]) == "dict[int, haystack.dataclasses.byte_stream.ByteStream]"
    # Chat Message
    assert serialize_type(list[ChatMessage]) == "list[haystack.dataclasses.chat_message.ChatMessage]"
    assert serialize_type(dict[int, ChatMessage]) == "dict[int, haystack.dataclasses.chat_message.ChatMessage]"
    # Document
    assert serialize_type(list[Document]) == "list[haystack.dataclasses.document.Document]"
    assert serialize_type(dict[int, Document]) == "dict[int, haystack.dataclasses.document.Document]"


def test_output_type_deserialization_haystack_dataclasses():
    # typing
    # Answer
    assert deserialize_type("haystack.dataclasses.answer.Answer") == Answer
    assert deserialize_type("typing.List[haystack.dataclasses.answer.Answer]") == List[Answer]
    assert deserialize_type("typing.Dict[int, haystack.dataclasses.answer.Answer]") == typing.Dict[int, Answer]
    # ByteStream
    assert deserialize_type("haystack.dataclasses.byte_stream.ByteStream") == ByteStream
    assert deserialize_type("typing.List[haystack.dataclasses.byte_stream.ByteStream]") == List[ByteStream]
    assert (
        deserialize_type("typing.Dict[int, haystack.dataclasses.byte_stream.ByteStream]")
        == typing.Dict[int, ByteStream]
    )
    # Chat Message
    assert deserialize_type("typing.List[haystack.dataclasses.chat_message.ChatMessage]") == typing.List[ChatMessage]
    assert (
        deserialize_type("typing.Dict[int, haystack.dataclasses.chat_message.ChatMessage]")
        == typing.Dict[int, ChatMessage]
    )
    assert deserialize_type("haystack.dataclasses.chat_message.ChatMessage") == ChatMessage
    # Document
    assert deserialize_type("haystack.dataclasses.document.Document") == Document
    assert deserialize_type("typing.List[haystack.dataclasses.document.Document]") == typing.List[Document]
    assert deserialize_type("typing.Dict[int, haystack.dataclasses.document.Document]") == typing.Dict[int, Document]
    # builtins
    # Answer
    assert deserialize_type("list[haystack.dataclasses.answer.Answer]") == list[Answer]
    assert deserialize_type("dict[int, haystack.dataclasses.answer.Answer]") == dict[int, Answer]
    # ByteStream
    assert deserialize_type("list[haystack.dataclasses.byte_stream.ByteStream]") == list[ByteStream]
    assert deserialize_type("dict[int, haystack.dataclasses.byte_stream.ByteStream]") == dict[int, ByteStream]
    # Chat Message
    assert deserialize_type("list[haystack.dataclasses.chat_message.ChatMessage]") == list[ChatMessage]
    assert deserialize_type("dict[int, haystack.dataclasses.chat_message.ChatMessage]") == dict[int, ChatMessage]
    # Document
    assert deserialize_type("list[haystack.dataclasses.document.Document]") == list[Document]
    assert deserialize_type("dict[int, haystack.dataclasses.document.Document]") == dict[int, Document]


def test_output_type_serialization_pep_604():
    # PEP 604 allows for union types to be defined with the `|` operator
    assert serialize_type(str | int) == "str | int"
    assert serialize_type(str | None) == "str | None"
    assert serialize_type(list[str] | None) == "list[str] | None"
    assert serialize_type(int | float | str) == "int | float | str"
    assert serialize_type(dict[str, int] | None) == "dict[str, int] | None"
    assert serialize_type(set[int] | None) == "set[int] | None"
    assert serialize_type(tuple[int, str] | None) == "tuple[int, str] | None"
    assert serialize_type(list[int] | list[str]) == "list[int] | list[str]"
    assert serialize_type(dict[str, int] | dict[int, str]) == "dict[str, int] | dict[int, str]"


def test_output_type_deserialization_pep_604():
    assert deserialize_type("str | int") == Union[str, int]
    assert deserialize_type("str | None") == Union[str, None]
    assert deserialize_type("int | float") == Union[int, float]
    assert deserialize_type("str | int | float") == Union[str, int, float]
    assert deserialize_type("str | int | None") == Union[str, int, None]
    assert deserialize_type("list[str] | None") == Union[list[str], None]
    assert deserialize_type("list[str] | list[int]") == Union[list[str], list[int]]
    assert deserialize_type("dict[str, int] | None") == Union[dict[str, int], None]
    assert deserialize_type("list[dict[str, int]] | None") == Union[list[dict[str, int]], None]
    assert deserialize_type("dict[str, list[int]] | set[str]") == Union[dict[str, list[int]], set[str]]
    assert deserialize_type("typing.List[str] | None") == Union[List[str], None]
    assert deserialize_type("typing.Dict[str, int] | typing.List[str]") == Union[Dict[str, int], List[str]]
    assert deserialize_type("set[int] | None") == Union[set[int], None]
    assert deserialize_type("tuple[int, str] | None") == Union[tuple[int, str], None]
    assert deserialize_type("frozenset[int] | None") == Union[frozenset[int], None]
    assert deserialize_type("dict[str, int] | dict[int, str]") == Union[dict[str, int], dict[int, str]]
    assert deserialize_type("list[int] | list[str] | list[float]") == Union[list[int], list[str], list[float]]


def test_is_union_type():
    assert _is_union_type(Union) is True
    assert _is_union_type(UnionType) is True
    assert _is_union_type(Union[str, int]) is True
    assert _is_union_type(str | int) is True
    assert _is_union_type(str | None) is True
    assert _is_union_type(Optional[str]) is True

    assert _is_union_type(str) is False
    assert _is_union_type(None) is False
    assert _is_union_type(list[str]) is False
    assert _is_union_type(dict[str, int]) is False


def test_parse_pep604_union_args():
    assert _parse_pep604_union_args("str | int") == ["str", "int"]
    assert _parse_pep604_union_args("str | None") == ["str", "None"]
    assert _parse_pep604_union_args("str | int | float") == ["str", "int", "float"]
    assert _parse_pep604_union_args("str | int | None") == ["str", "int", "None"]

    # Nested generics
    assert _parse_pep604_union_args("list[str] | None") == ["list[str]", "None"]
    assert _parse_pep604_union_args("list[str] | dict[str, int]") == ["list[str]", "dict[str, int]"]
    assert _parse_pep604_union_args("list[str] | dict[str, int] | None") == ["list[str]", "dict[str, int]", "None"]
    assert _parse_pep604_union_args("set[int] | None") == ["set[int]", "None"]
    assert _parse_pep604_union_args("tuple[int, str] | None") == ["tuple[int, str]", "None"]
    assert _parse_pep604_union_args("dict[str, list[int]] | set[str]") == ["dict[str, list[int]]", "set[str]"]
    assert _parse_pep604_union_args("list[int] | list[str] | list[float]") == ["list[int]", "list[str]", "list[float]"]


def test_build_pep604_union_type():
    result = _build_pep604_union_type([str])
    assert result == str

    result = _build_pep604_union_type([str, int])
    assert result == str | int

    result = _build_pep604_union_type([str, int, float])
    assert result == str | int | float

    result = _build_pep604_union_type([str, type(None)])
    assert result == str | None

    result = _build_pep604_union_type([list[str], dict[str, int]])
    assert result == list[str] | dict[str, int]


if sys.version_info < (3, 14):

    def test_type_de_se_union_and_optional():
        """Tests for old typing.Union and typing.Optional types that are converted to builtins in python 3.14+."""
        assert serialize_type(List[Union[str, int]]) == "typing.List[typing.Union[str, int]]"
        assert serialize_type(List[str] | List[int]) == "typing.Union[typing.List[str], typing.List[int]]"
        assert serialize_type(List[Optional[str]]) == "typing.List[typing.Optional[str]]"
        assert (
            serialize_type(Dict[str, int] | Dict[int, str])
            == "typing.Union[typing.Dict[str, int], typing.Dict[int, str]]"
        )
        assert serialize_type(list[Union[str, int]]) == "list[typing.Union[str, int]]"
        assert serialize_type(list[Optional[str]]) == "list[typing.Optional[str]]"
        # Union
        assert serialize_type(Union) == "typing.Union"
        assert serialize_type(Union[str, int]) == "typing.Union[str, int]"
        assert serialize_type(Union[int, float]) == "typing.Union[int, float]"
        assert serialize_type(Union[dict, str]) == "typing.Union[dict, str]"
        assert serialize_type(Union[float, bool]) == "typing.Union[float, bool]"
        assert serialize_type(Optional) == "typing.Optional"
        assert serialize_type(Optional[str]) == "typing.Optional[str]"
        assert serialize_type(Optional[int]) == "typing.Optional[int]"
        assert serialize_type(Optional[dict]) == "typing.Optional[dict]"
        assert serialize_type(Optional[float]) == "typing.Optional[float]"
        assert serialize_type(Optional[bool]) == "typing.Optional[bool]"


# `Annotated[T, m1, m2, ...]` holds a type T followed by metadata values. The metadata values are not
# types and must be rendered with repr() (so strings keep their quotes) and parsed with ast.literal_eval
# on the deserialize side. The split is quote-aware so a comma inside a string metadata value does not
# break the parse. Type-like metadata (classes, typing forms) is rendered through serialize_type and
# resolved through deserialize_type on the read side. This mirrors the Literal fix from PR #12286
# (commit 1f460e620) and the Callable-with-parameter-list fix from PR #12122.
def test_output_type_serialization_annotated():
    # String metadata — the failing case before the fix (silently serialized as `typing.Annotated[int, doc]`
    # and rejected on deserialize).
    assert serialize_type(Annotated[int, "doc"]) == "typing.Annotated[int, 'doc']"
    assert serialize_type(Annotated[str, "x", "y"]) == "typing.Annotated[str, 'x', 'y']"
    # Non-string literal metadata (int, bool, None, bytes) — the repr() of these is a valid Python
    # literal, so it round-trips through ast.literal_eval on the deserialize side.
    assert serialize_type(Annotated[int, 42]) == "typing.Annotated[int, 42]"
    assert serialize_type(Annotated[int, True]) == "typing.Annotated[int, True]"
    assert serialize_type(Annotated[int, None]) == "typing.Annotated[int, None]"
    assert serialize_type(Annotated[int, b"bytes"]) == "typing.Annotated[int, b'bytes']"
    # A comma inside a string metadata value must be preserved verbatim, not split.
    assert serialize_type(Annotated[int, "a, b"]) == "typing.Annotated[int, 'a, b']"
    # Ellipsis metadata — `...` is serialized as the literal `...` (same as the top-level Ellipsis
    # handling in serialize_type), so it round-trips through the existing Ellipsis handling on deserialize.
    assert serialize_type(Annotated[int, ...]) == "typing.Annotated[int, ...]"
    # Type metadata — a class is rendered with its module path (or bare name for builtins), not via repr.
    # `repr(str)` would be `<class 'str'>` which is not a valid Python literal and would not round-trip.
    assert serialize_type(Annotated[int, str]) == "typing.Annotated[int, str]"
    # A typing form as metadata is rendered as a type so nested generics and module paths are preserved.
    assert serialize_type(Annotated[int, List[str]]) == "typing.Annotated[int, typing.List[str]]"
    # A Literal as metadata is rendered as a type.
    assert serialize_type(Annotated[int, Literal["a", "b"]]) == "typing.Annotated[int, typing.Literal['a', 'b']]"
    # Nested in a generic — the inner Annotated is itself serialized through the Annotated branch.
    assert serialize_type(List[Annotated[int, "tag"]]) == "typing.List[typing.Annotated[int, 'tag']]"
    # Nested in an Optional — Python normalizes `Optional[X]` to `Union[X, None]`, so the wire form is
    # `typing.Optional[typing.Annotated[...]]` (the trailing None is dropped, same as for any other generic).
    assert serialize_type(Optional[Annotated[int, "doc"]]) == "typing.Optional[typing.Annotated[int, 'doc']]"
    # The wrapped type can itself be a generic with a top-level comma (Callable's parameter list).
    # The quote-aware split keeps the comma inside `Callable[[int, str], bool]` from being treated as
    # the Annotated separator.
    assert (
        serialize_type(Annotated[Callable[[int, str], bool], "doc"])
        == "typing.Annotated[typing.Callable[[int, str], bool], 'doc']"
    )


def test_output_type_deserialization_annotated():
    # String metadata round-trips.
    assert deserialize_type("typing.Annotated[int, 'doc']") == Annotated[int, "doc"]
    assert deserialize_type("typing.Annotated[str, 'x', 'y']") == Annotated[str, "x", "y"]
    # Non-string literal metadata.
    assert deserialize_type("typing.Annotated[int, 42]") == Annotated[int, 42]
    assert deserialize_type("typing.Annotated[int, True]") == Annotated[int, True]
    assert deserialize_type("typing.Annotated[int, None]") == Annotated[int, None]
    assert deserialize_type("typing.Annotated[int, b'bytes']") == Annotated[int, b"bytes"]
    # A comma inside a string metadata value must not split the args (quote-aware split).
    assert deserialize_type("typing.Annotated[int, 'a, b']") == Annotated[int, "a, b"]
    # Ellipsis metadata round-trips through the existing Ellipsis handling.
    assert deserialize_type("typing.Annotated[int, ...]") == Annotated[int, ...]
    # Type metadata — a class is resolved through the type-import path.
    assert deserialize_type("typing.Annotated[int, str]") == Annotated[int, str]
    # A typing form as metadata is resolved through deserialize_type.
    assert deserialize_type("typing.Annotated[int, typing.List[str]]") == Annotated[int, List[str]]
    # A Literal as metadata.
    assert deserialize_type("typing.Annotated[int, typing.Literal['a', 'b']]") == Annotated[int, Literal["a", "b"]]
    # Nested in a generic.
    assert deserialize_type("typing.List[typing.Annotated[int, 'tag']]") == List[Annotated[int, "tag"]]
    # Nested in an Optional.
    assert deserialize_type("typing.Optional[typing.Annotated[int, 'doc']]") == Optional[Annotated[int, "doc"]]
    # The wrapped type is a generic with a top-level comma (Callable's parameter list).
    assert (
        deserialize_type("typing.Annotated[typing.Callable[[int, str], bool], 'doc']")
        == Annotated[Callable[[int, str], bool], "doc"]
    )


def test_output_type_round_trip_annotated():
    # Round-trip all the kinds of metadata Python's Annotated accepts and that survive a text round-trip.
    # Mirrors the existing Literal trio (serialization, deserialization, round_trip).
    cases = [
        Annotated[int, "doc"],
        Annotated[str, "x", "y"],
        Annotated[int, 42],
        Annotated[int, True],
        Annotated[int, None],
        Annotated[int, b"bytes"],
        Annotated[int, "a, b"],  # comma in string metadata — quote-aware split
        Annotated[int, ...],  # Ellipsis metadata
        Annotated[int, str],  # class metadata
        Annotated[int, int],  # class metadata (built-in)
        Annotated[int, List[str]],  # typing form metadata
        Annotated[int, Literal["a", "b"]],  # Literal as metadata
        List[Annotated[int, "tag"]],  # nested in a generic
        Optional[Annotated[int, "doc"]],  # nested in Optional
        Annotated[Callable[[int, str], bool], "doc"],  # wrapped type has a top-level comma
    ]
    for type_ in cases:
        assert deserialize_type(serialize_type(type_)) == type_


def test_split_annotated_args():
    # The split is on the first top-level comma (outside brackets and outside string literals), so a
    # comma inside a string metadata value is not treated as the separator.
    from haystack.utils.type_serialization import _split_annotated_args

    assert _split_annotated_args("int, 'doc'") == ("int", "'doc'")
    assert _split_annotated_args("int, 'a, b'") == ("int", "'a, b'")
    # The wrapped type can contain brackets and commas inside them (e.g. Callable's parameter list):
    # the comma inside the brackets is at depth > 0 and is not the separator.
    assert _split_annotated_args("typing.Callable[[int, str], bool], 'doc'") == (
        "typing.Callable[[int, str], bool]",
        "'doc'",
    )
    # Double-quoted metadata works the same way.
    assert _split_annotated_args('int, "doc"') == ("int", '"doc"')
    # No top-level comma: the whole string is the type and there is no metadata (defensive: Annotated
    # always has metadata in practice, since Python normalizes the bare Annotated[T] form back to T).
    assert _split_annotated_args("int") == ("int", "")


def test_output_type_deserialization_annotated_no_metadata_errors():
    # The wire form `typing.Annotated[int]` is not a valid Annotated (Python normalizes the bare form
    # back to `int`), so an empty metadata chunk on the deserialize side is a hard error rather than a
    # silent misshape.
    with pytest.raises(DeserializationError, match="Annotated requires at least one metadata value"):
        deserialize_type("typing.Annotated[int]")
