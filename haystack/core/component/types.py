# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from dataclasses import dataclass, field
from types import UnionType
from typing import Annotated, Any, TypeAlias, TypedDict, TypeVar, get_args

HAYSTACK_VARIADIC_ANNOTATION = "__haystack__variadic_t"
HAYSTACK_GREEDY_VARIADIC_ANNOTATION = "__haystack__greedy_variadic_t"

# # Generic type variable used in the Variadic container
T = TypeVar("T")


# Variadic is a custom annotation type we use to mark input types.
# This type doesn't do anything else than "marking" the contained
# type so it can be used in the `InputSocket` creation where we
# check that its annotation equals to HAYSTACK_VARIADIC_ANNOTATION
Variadic: TypeAlias = Annotated[Iterable[T], HAYSTACK_VARIADIC_ANNOTATION]

# GreedyVariadic type is similar to Variadic.
# The only difference is the way it's treated by the Pipeline when input is received
# in a socket with this type.
# Instead of waiting for other inputs to be received, Components that have a GreedyVariadic
# input will be run right after receiving the first input.
# Even if there are multiple connections to that socket.
GreedyVariadic: TypeAlias = Annotated[Iterable[T], HAYSTACK_GREEDY_VARIADIC_ANNOTATION]


class _empty:
    """Custom object for marking InputSocket.default_value as not set."""


@dataclass
class InputSocket:
    """
    Represents an input of a `Component`.

    :param name:
        The name of the input.
    :param type:
        The type of the input.
    :param default_value:
        The default value of the input. If not set, the input is mandatory.
    :param is_lazy_variadic:
        Whether the input is a lazy variadic or not.
    :param is_greedy:
        Whether the input is a greedy variadic or not.
    :param senders:
        The list of components that send data to this input.
    :param wrap_input_in_list:
        Whether to wrap the input in a list before passing it to the component.
        Only applies to lazy variadic inputs so when is_lazy_variadic is True.
    """

    name: str
    type: type | UnionType
    default_value: Any = _empty
    is_lazy_variadic: bool = field(init=False)
    is_greedy: bool = field(init=False)
    senders: list[str] = field(default_factory=list)
    wrap_input_in_list: bool = True

    @property
    def is_variadic(self):
        """Check if the input is variadic."""
        return self.is_greedy or self.is_lazy_variadic

    @property
    def is_mandatory(self):
        """Check if the input is mandatory."""
        return self.default_value == _empty

    def __post_init__(self):
        try:
            # __metadata__ is a tuple
            self.is_lazy_variadic = (
                hasattr(self.type, "__metadata__") and self.type.__metadata__[0] == HAYSTACK_VARIADIC_ANNOTATION
            )
            self.is_greedy = (
                hasattr(self.type, "__metadata__") and self.type.__metadata__[0] == HAYSTACK_GREEDY_VARIADIC_ANNOTATION
            )
        except AttributeError:
            self.is_lazy_variadic = False
            self.is_greedy = False

        # We need to "unpack" the type inside the Variadic annotation, otherwise the pipeline connection api will try
        # to match `Annotated[type, HAYSTACK_VARIADIC_ANNOTATION]`.
        #
        # Note1: Variadic is expressed as an annotation of one single type, so the return value of get_args will
        # always be a one-item tuple.
        #
        # Note2: a pipeline always passes a list of items when a component input is declared as Variadic, so the
        # type itself always wraps an iterable of the declared type. For example, Variadic[int] is eventually an
        # alias for Iterable[int]. Since we're interested in getting the inner type `int`, we call `get_args`
        # twice: the first time to get `list[int]` out of `Variadic`, the second time to get `int` out of `list[int]`.
        if self.is_lazy_variadic or self.is_greedy:
            self.type = get_args(get_args(self.type)[0])[0]


class InputSocketTypeDescriptor(TypedDict):
    """
    Describes the type of `InputSocket`.
    """

    type: type | UnionType
    is_mandatory: bool


@dataclass
class OutputSocket:
    """
    Represents an output of a `Component`.

    :param name:
        The name of the output.
    :param type:
        The type of the output.
    :param receivers:
        The list of components that receive the output of this component.
    """

    name: str
    type: type
    receivers: list[str] = field(default_factory=list)
