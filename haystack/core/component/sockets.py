# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import get_args, List, Type, Any
import logging
from dataclasses import dataclass, field

from haystack.core.component.types import CANALS_VARIADIC_ANNOTATION


logger = logging.getLogger(__name__)


class _empty:
    """Custom object for marking InputSocket.default_value as not set."""


@dataclass
class InputSocket:
    name: str
    type: Type
    default_value: Any = _empty
    is_variadic: bool = field(init=False)
    senders: List[str] = field(default_factory=list)

    @property
    def is_mandatory(self):
        return self.default_value == _empty

    def __post_init__(self):
        try:
            # __metadata__ is a tuple
            self.is_variadic = self.type.__metadata__[0] == CANALS_VARIADIC_ANNOTATION
        except AttributeError:
            self.is_variadic = False
        if self.is_variadic:
            # We need to "unpack" the type inside the Variadic annotation,
            # otherwise the pipeline connection api will try to match
            # `Annotated[type, CANALS_VARIADIC_ANNOTATION]`.
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
    name: str
    type: type
    receivers: List[str] = field(default_factory=list)
