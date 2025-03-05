# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Annotated, Any, Iterable, List, Type, TypeVar, get_args

from typing_extensions import TypeAlias  # Python 3.9 compatibility

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
    :param is_variadic:
        Whether the input is variadic or not.
    :param is_greedy
        Whether the input is a greedy variadic or not.
    :param senders:
        The list of components that send data to this input.
    """

    name: str
    type: Type
    default_value: Any = _empty
    is_variadic: bool = field(init=False)
    is_greedy: bool = field(init=False)
    senders: List[str] = field(default_factory=list)

    @property
    def is_mandatory(self):
        """Check if the input is mandatory."""
        return self.default_value == _empty

    def __post_init__(self):
        try:
            # __metadata__ is a tuple
            self.is_variadic = self.type.__metadata__[0] in [
                HAYSTACK_VARIADIC_ANNOTATION,
                HAYSTACK_GREEDY_VARIADIC_ANNOTATION,
            ]
            self.is_greedy = self.type.__metadata__[0] == HAYSTACK_GREEDY_VARIADIC_ANNOTATION
        except AttributeError:
            self.is_variadic = False
            self.is_greedy = False
        if self.is_variadic:
            # We need to "unpack" the type inside the Variadic annotation,
            # otherwise the pipeline connection api will try to match
            # `Annotated[type, HAYSTACK_VARIADIC_ANNOTATION]`.
            #
            # Note1: Variadic is expressed as an annotation of one single type,
            # so the return value of get_args will always be a one-item tuple.
            #
            # Note2: a pipeline always passes a list of items when a component
            # input is declared as Variadic, so the type itself always wraps
            # an iterable of the declared type. For example, Variadic[int]
            # is eventually an alias for Iterable[int]. Since we're interested
            # in getting the inner type `int`, we call `get_args` twice: the
            # first time to get `List[int]` out of `Variadic`, the second time
            # to get `int` out of `List[int]`.
            self.type = get_args(get_args(self.type)[0])[0]


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
    receivers: List[str] = field(default_factory=list)
