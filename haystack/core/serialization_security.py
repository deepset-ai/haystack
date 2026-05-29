# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Security primitives for pipeline deserialization.

This module provides an allowlist mechanism that gates arbitrary imports.

Three ways to extend the allowlist:
- Per-call kwarg: `Pipeline.load(..., allowed_modules=["mypkg.*"])`
- Process-wide programmatic API: :func:`allow_deserialization_module`
- Environment variable: `HAYSTACK_DESERIALIZATION_ALLOWLIST="mypkg.*,otherpkg.*"`

The two-mode loading API (`unsafe=True`) bypasses the allowlist entirely.
"""

import contextvars
import fnmatch
import os
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from haystack.core.errors import DeserializationError

# The default allowlist covers Haystack's own packages plus a small set of standard-library type modules
# that are commonly referenced in serialized type annotations (e.g. `typing.List[str]`,
# `collections.deque`). Importing these modules has no meaningful side effects on its own.
DEFAULT_ALLOWED_MODULES: tuple[str, ...] = (
    "haystack",
    "haystack_integrations",
    "haystack_experimental",
    "builtins",
    "typing",
    "collections",
)
DESERIALIZATION_ALLOWLIST_ENV_VAR = "HAYSTACK_DESERIALIZATION_ALLOWLIST"


@dataclass(frozen=True)
class _DeserializationContext:
    extra_allowed: tuple[str, ...] = field(default_factory=tuple)
    unsafe: bool = False


_current_context: contextvars.ContextVar[_DeserializationContext | None] = contextvars.ContextVar(
    "haystack_deserialization_context", default=None
)


def _get_context() -> _DeserializationContext:
    ctx = _current_context.get()
    return ctx if ctx is not None else _DeserializationContext()


# Process-wide patterns set via allow_deserialization_module.
_extra_allowed_modules: list[str] = []


def allow_deserialization_module(pattern: str) -> None:
    """
    Add a module pattern to the process-wide deserialization allowlist.

    Once added, classes from modules matching the pattern can be deserialized from YAML / dict
    representations until the process exits.

    A pattern matches a module name if:
    - The pattern contains `*`, `?` or `[` — :mod:`fnmatch` semantics are used.
    - Otherwise the pattern is treated as a prefix: a module matches if it equals the pattern or
      is a submodule of it (i.e. starts with `pattern + "."`). A trailing `.*` is stripped
      before this comparison, so `"mypkg"` and `"mypkg.*"` behave identically.

    :param pattern:
        The module pattern to allow.
    """
    if pattern not in _extra_allowed_modules:
        _extra_allowed_modules.append(pattern)


def _module_matches(module_name: str, pattern: str) -> bool:
    """Return whether `module_name` matches the given allowlist `pattern`."""
    # `pkg.*` is treated as a prefix match (matches `pkg` and any submodule); this is the most
    # common form, and we want it to match the bare top-level package too, which fnmatch wouldn't.
    if pattern.endswith(".*"):
        pattern = pattern[:-2]
        return module_name == pattern or module_name.startswith(pattern + ".")
    if any(c in pattern for c in "*?["):
        return fnmatch.fnmatchcase(module_name, pattern)
    return module_name == pattern or module_name.startswith(pattern + ".")


def _patterns_from_env() -> list[str]:
    raw = os.environ.get(DESERIALIZATION_ALLOWLIST_ENV_VAR, "")
    return [p.strip() for p in raw.split(",") if p.strip()]


def _is_module_allowed(module_name: str) -> bool:
    """Return whether `module_name` is on the active deserialization allowlist."""
    ctx = _get_context()
    if ctx.unsafe:
        return True
    patterns: list[str] = []
    patterns.extend(DEFAULT_ALLOWED_MODULES)
    patterns.extend(_extra_allowed_modules)
    patterns.extend(_patterns_from_env())
    patterns.extend(ctx.extra_allowed)
    return any(_module_matches(module_name, p) for p in patterns)


def _check_module_allowed(module_name: str) -> None:
    """Raise :class:`DeserializationError` if `module_name` is not on the allowlist."""
    if _is_module_allowed(module_name):
        return
    raise DeserializationError(
        f"Refusing to deserialize a class from module '{module_name}': the module is not on the "
        f"trusted-module allowlist. If you trust the source of this serialized data, you can either:\n"
        f"  - extend the allowlist for this call: "
        f"Pipeline.load(..., allowed_modules=['{module_name}']),\n"
        f"  - extend it process-wide via haystack.core.serialization.allow_deserialization_module"
        f"('{module_name}') or the {DESERIALIZATION_ALLOWLIST_ENV_VAR} environment variable,\n"
        f"  - or bypass the allowlist entirely: Pipeline.load(..., unsafe=True)."
    )


@contextmanager
def _deserialization_context(allowed_modules: Iterable[str] | None = None, unsafe: bool = False) -> Iterator[None]:
    """
    Context manager that activates a per-call deserialization context.

    Patterns from `allowed_modules` are appended to the parent context's patterns, and `unsafe`
    is OR-ed with the parent's `unsafe` flag — so this never narrows the active permissions.
    The previous context is restored on exit.
    """
    parent = _get_context()
    extra = parent.extra_allowed + (tuple(allowed_modules) if allowed_modules else ())
    merged_unsafe = parent.unsafe or unsafe
    token = _current_context.set(_DeserializationContext(extra_allowed=extra, unsafe=merged_unsafe))
    try:
        yield
    finally:
        _current_context.reset(token)
