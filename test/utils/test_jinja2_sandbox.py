# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os

import jinja2
import pytest

from haystack.utils.jinja2_sandbox import HaystackSandboxedEnvironment


class TestHaystackSandboxedEnvironment:
    def test_blocks_module_attribute_access(self):
        # Reaching into a module object (e.g. os.system) is the final step of the reported escape.
        env = HaystackSandboxedEnvironment()
        with pytest.raises(jinja2.exceptions.SecurityError):
            env.from_string("{{ mod.system('echo pwned') }}").render(mod=os)

    def test_blocks_calling_dangerous_module_callable(self):
        env = HaystackSandboxedEnvironment()
        with pytest.raises(jinja2.exceptions.SecurityError):
            env.from_string("{{ fn('echo pwned') }}").render(fn=os.system)

    def test_blocks_calling_module_object(self):
        env = HaystackSandboxedEnvironment()
        with pytest.raises(jinja2.exceptions.SecurityError):
            env.from_string("{{ mod() }}").render(mod=os)

    def test_allows_builtin_string_methods(self):
        # `builtins` is intentionally excluded from the callable blocklist so ordinary template
        # operations keep working.
        env = HaystackSandboxedEnvironment()
        assert env.from_string("{{ name.upper() }}").render(name="hi") == "HI"

    def test_allows_custom_filter(self):
        # Filters are invoked directly by Jinja and are unaffected by the sandbox hardening.
        env = HaystackSandboxedEnvironment()
        env.filters["shout"] = lambda v: v.upper()
        assert env.from_string("{{ name | shout }}").render(name="hi") == "HI"

    def test_allows_object_data_access(self):
        env = HaystackSandboxedEnvironment()
        assert env.from_string("{{ doc['content'] }}").render(doc={"content": "hello"}) == "hello"
