# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os

import jinja2
import pytest

from haystack.dataclasses import ByteStream, Document
from haystack.utils.auth import Secret
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

    def test_allows_document_attribute_access(self):
        # Plain attribute access on a Haystack data class must keep working; only calling its
        # methods is restricted.
        env = HaystackSandboxedEnvironment()
        doc = Document(content="hello", meta={"source": "handbook"})
        assert env.from_string("{{ doc.content }} {{ doc.meta.source }}").render(doc=doc) == "hello handbook"

    def test_blocks_document_from_dict_gadget(self, tmp_path):
        # Document.from_dict()/ByteStream.to_file() let a template write an arbitrary file.
        env = HaystackSandboxedEnvironment()
        doc = Document(content="hello")
        target = tmp_path / "pwned"
        template = (
            '{{ doc.from_dict({"content": "x", "blob": {"data": [104, 105], "meta": {}}})'
            f'.blob.to_file("{target}") }}}}'
        )
        with pytest.raises(jinja2.exceptions.SecurityError):
            env.from_string(template).render(doc=doc)
        assert not target.exists()

    def test_blocks_bytestream_from_file_path_gadget(self):
        # ByteStream.from_file_path() lets a template read an arbitrary file into the rendered output.
        env = HaystackSandboxedEnvironment()
        stream = ByteStream(data=b"")
        with pytest.raises(jinja2.exceptions.SecurityError):
            env.from_string('{{ stream.from_file_path("/etc/passwd").to_string() }}').render(stream=stream)

    def test_blocks_secret_resolve_value_gadget(self, monkeypatch):
        # Secret.resolve_value() lets a template read arbitrary environment variables.
        monkeypatch.setenv("HAYSTACK_TEST_SECRET", "super-secret")
        env = HaystackSandboxedEnvironment()
        secret = Secret.from_env_var("HAYSTACK_TEST_SECRET")
        with pytest.raises(jinja2.exceptions.SecurityError):
            env.from_string("{{ secret.resolve_value() }}").render(secret=secret)
