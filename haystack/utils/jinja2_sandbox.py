# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from types import ModuleType
from typing import Any

from jinja2.sandbox import SandboxedEnvironment

# Roots of modules whose callables must never be invocable from a template. Reaching one of these
# from inside a rendered template is only possible after a sandbox escape (e.g. a custom filter that
# returns a module object), so blocking the call is a defense-in-depth backstop. `builtins` is
# deliberately excluded: ordinary template operations such as `{{ name.upper() }}` resolve to
# builtin methods, and blocking those would break legitimate templates.
_UNSAFE_MODULE_ROOTS: frozenset[str] = frozenset(
    {
        "os",
        "sys",
        "subprocess",
        "socket",
        "shutil",
        "importlib",
        "ctypes",
        "posix",
        "nt",
        "pty",
        "pickle",
        "shelve",
        "marshal",
        "multiprocessing",
        "code",
        "pdb",
    }
)

# Full module prefixes whose callables must never be invocable from a template

# Haystack's own data classes, which are in the template context (e.g. `documents`, `messages`) and expose public
# methods that perform file I/O or resolve secrets (`ByteStream.to_file`, `ByteStream.from_file_path`,
# `Document.from_dict`, `Secret.resolve_value`, ...).
#
# Templates only ever need plain attribute access on these objects (`doc.content`, `doc.meta`), never their methods,
# so calls into these modules are denied outright rather than allowlisted method by method.
_UNSAFE_CALLABLE_MODULE_PREFIXES: tuple[str, ...] = ("haystack.dataclasses", "haystack.utils.auth")


class HaystackSandboxedEnvironment(SandboxedEnvironment):
    """
    A `SandboxedEnvironment` hardened against sandbox-escape gadgets.

    On top of Jinja2's stock sandbox it additionally:

    - refuses attribute access on module objects, so a module that leaks into the template context
      (e.g. via a custom filter that imports one) cannot be walked into (`os.system`, ...);
    - refuses to call module objects, refuses to call any callable whose defining module is rooted in
      a dangerous standard-library module (see :data:`_UNSAFE_MODULE_ROOTS`), and refuses to call any
      callable defined in one of Haystack's own data-class modules (see
      :data:`_UNSAFE_CALLABLE_MODULE_PREFIXES`).

    Note that Jinja invokes *filters* directly, bypassing `is_safe_callable`, so this does not
    constrain what a registered `custom_filters` function itself does; it only governs attribute
    access and calls written in template text.
    """

    def is_safe_attribute(self, obj: Any, attr: str, value: Any) -> bool:
        """Reject attribute access on module objects; otherwise defer to the stock sandbox."""
        # Templates never legitimately reach into a module object's attributes.
        if isinstance(obj, ModuleType):
            return False
        return super().is_safe_attribute(obj, attr, value)

    def is_safe_callable(self, obj: Any) -> bool:
        """Reject calling module objects, dangerous-module callables, and Haystack data-class methods."""
        if isinstance(obj, ModuleType):
            return False
        module = getattr(obj, "__module__", "") or ""
        root = module.split(".", 1)[0]
        if root in _UNSAFE_MODULE_ROOTS:
            return False
        if module.startswith(_UNSAFE_CALLABLE_MODULE_PREFIXES):
            return False
        return super().is_safe_callable(obj)
