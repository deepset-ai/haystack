from typing import TypeVar, Iterable
from typing_extensions import TypeAlias, Annotated  # Python 3.8 compatibility

CANALS_VARIADIC_ANNOTATION = "__canals__variadic_t"

# # Generic type variable used in the Variadic container
T = TypeVar("T")


# Variadic is a custom annotation type we use to mark input types.
# This type doesn't do anything else than "marking" the contained
# type so it can be used in the `InputSocket` creation where we
# check that its annotation equals to CANALS_VARIADIC_ANNOTATION
Variadic: TypeAlias = Annotated[Iterable[T], CANALS_VARIADIC_ANNOTATION]
