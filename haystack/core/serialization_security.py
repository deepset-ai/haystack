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

import builtins
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

# `builtins` is on the default allowlist because deserialization legitimately needs builtin *types*
# (e.g. `builtins.str`, used in serialized type annotations and as nested `{"type": ...}` class
# references) and harmless builtin callables that Haystack's own serializer emits (e.g.
# `serialize_callable(print)` -> `"builtins.print"`). The module-granular allowlist is too coarse
# to separate those from dangerous members, so the two builtin-resolving contexts are gated
# differently:
#   - Type / class contexts (`deserialize_type`, `import_class_by_name`) require the resolved
#     builtin to be a `type` (see :func:`_check_builtin_is_type`). That lets every builtin type
#     through while rejecting every builtin *function*, with no denylist to maintain.
#   - The callable context (`deserialize_callable`) genuinely returns functions, so it instead
#     rejects the dangerous builtin *callables* named below (see :func:`_check_not_denied_builtin`).
_DENIED_BUILTIN_NAMES: frozenset[str] = frozenset(
    {
        "eval",  # arbitrary code execution
        "exec",  # arbitrary code execution
        "compile",  # arbitrary code compilation
        "__import__",  # dynamic import of any module (gateway to os/subprocess/...)
        "open",  # filesystem read/write
        "getattr",  # attribute-traversal gadget (classic sandbox escape)
        "setattr",  # arbitrary attribute mutation
        "delattr",  # arbitrary attribute deletion
        "globals",  # access to module namespaces
        "locals",  # access to local namespaces
        "vars",  # access to object/module namespaces
        "breakpoint",  # runs the PYTHONBREAKPOINT hook
        "__build_class__",  # dynamic class creation
        "type",  # dynamic class creation via type(name, bases, dict)
    }
)

# Resolve names to objects once so callers can match by identity, which also catches aliases that
# reach the same builtin via a different import path (e.g. `io.open is builtins.open`).
_DENIED_BUILTIN_OBJECTS: frozenset = frozenset([getattr(builtins, name) for name in _DENIED_BUILTIN_NAMES])


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
    # `pkg.*` (where the part before `.*` has no other wildcards) is treated as a prefix match —
    # matches `pkg` and any submodule. This is the most common form, and we want it to match
    # the bare top-level package too (which true fnmatch wouldn't, since `pkg.*` requires a
    # literal `.` to follow). Patterns like `j*on.*` keep their wildcards and fall through to
    # fnmatch so the semantics stay consistent.
    if pattern.endswith(".*") and not any(c in pattern[:-2] for c in "*?["):
        prefix = pattern[:-2]
        return module_name == prefix or module_name.startswith(prefix + ".")
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


def _is_denied_builtin(resolved: object) -> bool:
    """
    Return whether `resolved` is one of the builtins denied for callable deserialization.

    Matches by identity (not membership) so an unhashable resolved object never raises.
    """
    return any(resolved is denied for denied in _DENIED_BUILTIN_OBJECTS)


def _check_not_denied_builtin(resolved: object, handle: str) -> None:
    """
    Reject `resolved` if it is a builtin callable that is unsafe to resolve from serialized data.

    Used by the callable-resolution path (`deserialize_callable`). Raises
    :class:`DeserializationError` for the primitives in :data:`_DENIED_BUILTIN_NAMES`, which can
    execute code, import modules, touch the filesystem, or escape via attribute/namespace access.
    The block applies even though `builtins` is on the allowlist, because the allowlist is
    module-granular. It is intentionally bypassed in `unsafe=True` mode, which disables all
    deserialization safety checks by design.

    :param resolved:
        The object resolved from the serialized handle.
    :param handle:
        The original serialized handle, used only for the error message.
    """
    if _get_context().unsafe:
        return
    if _is_denied_builtin(resolved):
        name = getattr(resolved, "__name__", str(resolved))
        raise DeserializationError(
            f"Refusing to deserialize '{handle}': it resolves to the builtin '{name}', which is "
            f"blocked because it can be used to execute code, import modules, access the "
            f"filesystem, or escape via attribute access. If you trust the source of this data, "
            f"load it with unsafe=True to bypass deserialization safety checks."
        )


def _check_builtin_is_type(resolved: object, handle: str) -> None:
    """
    Reject a `builtins` member resolved in a type/class context that is not a `type`.

    Used by `deserialize_type` and `import_class_by_name`, which resolve type annotations and class
    references — always classes. Requiring the resolved `builtins` member to be a `type` lets every
    builtin type through (e.g. `str`, `memoryview`) while rejecting every builtin *function* (e.g.
    `eval`, `exec`, `getattr`), with no denylist to maintain. Bypassed in `unsafe=True` mode.

    :param resolved:
        The object resolved from the serialized handle.
    :param handle:
        The original serialized handle, used only for the error message.
    """
    if _get_context().unsafe:
        return
    if not isinstance(resolved, type):
        raise DeserializationError(
            f"Refusing to deserialize '{handle}': it resolves to a builtin that is not a type and "
            f"cannot be used as a type annotation or class reference. If you trust the source of "
            f"this data, load it with unsafe=True to bypass deserialization safety checks."
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
