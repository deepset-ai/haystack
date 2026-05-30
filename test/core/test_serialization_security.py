# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import io
import json
import subprocess
from collections.abc import Callable

import pytest

from haystack import component as component_module
from haystack.core.errors import DeserializationError
from haystack.core.pipeline import Pipeline
from haystack.core.serialization import allow_deserialization_module, import_class_by_name
from haystack.core.serialization_security import (
    DESERIALIZATION_ALLOWLIST_ENV_VAR,
    _check_module_allowed,
    _current_context,
    _deserialization_context,
    _DeserializationContext,
    _extra_allowed_modules,
    _is_module_allowed,
    _module_matches,
)
from haystack.marshal import YamlMarshaller
from haystack.utils import deserialize_callable


@pytest.fixture(autouse=True)
def _reset_allowlist_state():
    """
    Force a clean (safe-default, no extra patterns) state for every test in this module so we are
    testing the actual security model. The top-level test conftest extends the process-wide
    allowlist with test-only patterns (`test_*`, `pydantic`, ...); we must clear those here so
    "untrusted" really means untrusted.
    """
    snapshot = list(_extra_allowed_modules)
    _extra_allowed_modules.clear()
    token = _current_context.set(_DeserializationContext())
    try:
        yield
    finally:
        _extra_allowed_modules.clear()
        _extra_allowed_modules.extend(snapshot)
        _current_context.reset(token)


class TestModuleMatches:
    def test_prefix_match_equal(self):
        assert _module_matches("haystack", "haystack")

    def test_prefix_match_submodule(self):
        assert _module_matches("haystack.components.builders", "haystack")

    def test_prefix_match_strips_trailing_wildcard(self):
        assert _module_matches("haystack.components", "haystack.*")
        assert _module_matches("haystack", "haystack.*")

    def test_prefix_match_not_a_partial_word(self):
        assert not _module_matches("haystack_other", "haystack")

    def test_trailing_star_matches_submodules(self):
        assert _module_matches("mypkg.components.foo", "mypkg.*")
        assert _module_matches("mypkg.foo.bar", "mypkg.*")

    def test_trailing_star_does_not_match_unrelated(self):
        assert not _module_matches("other.foo", "mypkg.*")

    def test_fnmatch_glob_in_middle(self):
        assert _module_matches("pkg.foo.utils", "pkg.*.utils")
        assert _module_matches("pkg.bar.utils", "pkg.*.utils")

    def test_fnmatch_glob_in_middle_no_match(self):
        assert not _module_matches("pkg.foo.helpers", "pkg.*.utils")

    def test_fnmatch_single_char(self):
        # `?` is an fnmatch wildcard for a single character.
        assert _module_matches("pkga", "pkg?")
        assert not _module_matches("pkgab", "pkg?")

    def test_fnmatch_character_class(self):
        assert _module_matches("data_3", "data_[0-9]")
        assert not _module_matches("data_x", "data_[0-9]")

    def test_trailing_star_with_wildcards_in_prefix_uses_fnmatch(self):
        # `j*on.*` has a `*` before the trailing `.*`, so it must NOT be short-circuited to a
        # prefix match against the literal `j*on`. It should fall through to fnmatch.
        assert _module_matches("json.tool", "j*on.*")
        assert _module_matches("jaeon.subpkg.foo", "j*on.*")
        # Pure fnmatch doesn't match the bare `json` for the pattern `j*on.*` (the `.*` requires
        # a `.X` part).
        assert not _module_matches("json", "j*on.*")


class TestAllowlistDefaults:
    def test_haystack_allowed(self):
        assert _is_module_allowed("haystack")
        assert _is_module_allowed("haystack.components.builders.prompt_builder")

    def test_haystack_integrations_allowed(self):
        assert _is_module_allowed("haystack_integrations.components.retrievers")

    def test_haystack_experimental_allowed(self):
        assert _is_module_allowed("haystack_experimental")

    def test_typing_allowed(self):
        assert _is_module_allowed("typing")

    def test_collections_allowed(self):
        assert _is_module_allowed("collections")
        assert _is_module_allowed("collections.abc")

    def test_builtins_allowed(self):
        assert _is_module_allowed("builtins")

    def test_arbitrary_third_party_not_allowed(self):
        assert not _is_module_allowed("subprocess")
        assert not _is_module_allowed("os")


class TestAllowDeserializationModule:
    def test_extends_allowlist(self):
        assert not _is_module_allowed("mypkg.components")
        allow_deserialization_module("mypkg")
        assert _is_module_allowed("mypkg")
        assert _is_module_allowed("mypkg.components")

    def test_pattern_with_wildcard(self):
        allow_deserialization_module("mypkg.components.*")
        assert _is_module_allowed("mypkg.components.foo")

    def test_duplicate_pattern_only_added_once(self):
        allow_deserialization_module("mypkg")
        allow_deserialization_module("mypkg")
        assert _extra_allowed_modules.count("mypkg") == 1


class TestDeserializationContext:
    def test_extra_allowed_modules_via_context(self):
        assert not _is_module_allowed("mypkg.thing")
        with _deserialization_context(allowed_modules=["mypkg"]):
            assert _is_module_allowed("mypkg.thing")
        # The per-call extension is reset on exit.
        assert not _is_module_allowed("mypkg.thing")

    def test_unsafe_bypasses_allowlist(self):
        assert not _is_module_allowed("subprocess")
        with _deserialization_context(unsafe=True):
            assert _is_module_allowed("subprocess")
            assert _is_module_allowed("any.arbitrary.module")
        assert not _is_module_allowed("subprocess")


class TestEnvVar:
    def test_env_var_extends_allowlist(self, monkeypatch):
        monkeypatch.setenv(DESERIALIZATION_ALLOWLIST_ENV_VAR, "mypkg.components.*,otherpkg")
        assert _is_module_allowed("mypkg.components.foo")
        assert _is_module_allowed("otherpkg")
        assert _is_module_allowed("otherpkg.sub")
        assert not _is_module_allowed("yetanother")

    def test_env_var_ignores_empty_entries(self, monkeypatch):
        monkeypatch.setenv(DESERIALIZATION_ALLOWLIST_ENV_VAR, ", ,mypkg,,")
        assert _is_module_allowed("mypkg.sub")


class TestCheckModuleAllowed:
    def test_passes_silently_for_allowed_module(self):
        _check_module_allowed("haystack.foo")

    def test_raises_for_disallowed_module(self):
        with pytest.raises(DeserializationError, match="not on the trusted-module allowlist"):
            _check_module_allowed("subprocess")

    def test_error_message_suggests_remediations(self):
        with pytest.raises(DeserializationError) as exc_info:
            _check_module_allowed("mypkg.evil")
        message = str(exc_info.value)
        assert "allowed_modules" in message
        assert "allow_deserialization_module" in message
        assert DESERIALIZATION_ALLOWLIST_ENV_VAR in message
        assert "unsafe=True" in message


class TestImportClassByNameAllowlist:
    def test_allowlisted_class(self):
        cls = import_class_by_name("haystack.core.pipeline.Pipeline")
        assert cls is Pipeline

    def test_rejects_untrusted_module(self):
        with pytest.raises(DeserializationError, match="not on the trusted-module allowlist"):
            import_class_by_name("subprocess.Popen")

    def test_per_call_extension(self):
        # subprocess is normally blocked
        with pytest.raises(DeserializationError):
            import_class_by_name("subprocess.Popen")
        # ... but extending the allowlist for a single call lets it through.
        with _deserialization_context(allowed_modules=["subprocess"]):
            cls = import_class_by_name("subprocess.Popen")
            assert cls is subprocess.Popen


class TestDeserializeCallableAllowlist:
    """
    `deserialize_callable` walks progressively-shorter module prefixes when resolving a dotted
    name. The allowlist check must apply to "is *any* prefix on the allowlist?", not to each
    individual candidate — otherwise fnmatch patterns that match the actual module but not the
    full handle (e.g. `j*on` matches `json` but not `json.dumps`) would be wrongly rejected.
    """

    def test_fnmatch_pattern_matches_shorter_prefix(self):
        # `j*on` matches `json` (the actual module) but not `json.dumps` (the full handle).
        # The deferred allowlist check should still accept this.
        with _deserialization_context(allowed_modules=["j*on"]):
            fn = deserialize_callable("json.dumps")
            assert fn is json.dumps

    def test_rejects_when_no_prefix_matches(self):
        # No prefix of `subprocess.Popen` matches the default allowlist (or `unrelated`).
        with _deserialization_context(allowed_modules=["unrelated"]):
            with pytest.raises(DeserializationError, match="not on the trusted-module allowlist"):
                deserialize_callable("subprocess.Popen")


@pytest.fixture
def _registered_untrusted_component():
    """
    Set up a fake component class registered under a fully-qualified name in an untrusted module
    (`evilpkg.evilmod.EvilComponent`). Yields a dict payload referencing it. The fixture cleans
    up the registry on teardown.
    """
    fake_type = "evilpkg.evilmod.EvilComponent"

    @component_module
    class EvilComponent:
        @component_module.output_types(value=int)
        def run(self, value: int) -> dict[str, int]:
            return {"value": value}

    registry = component_module.registry
    original = registry.get(fake_type)
    registry[fake_type] = EvilComponent
    try:
        yield {
            "fake_type": fake_type,
            "data": {
                "metadata": {},
                "components": {"evil": {"type": fake_type, "init_parameters": {}}},
                "connections": [],
            },
        }
    finally:
        if original is None:
            registry.pop(fake_type, None)
        else:
            registry[fake_type] = original


class TestPipelineFromDictAllowlistBypass:
    def test_pre_registered_untrusted_component_is_rejected(self, _registered_untrusted_component):
        with pytest.raises(DeserializationError, match="not on the trusted-module allowlist"):
            Pipeline.from_dict(_registered_untrusted_component["data"])

    def test_pre_registered_component_loadable_with_allowed_modules(self, _registered_untrusted_component):
        """
        Counterpart to the bypass test: once the user opts the module into the allowlist, the
        load gets past the allowlist gate. (It still fails downstream because the fake type name
        doesn't match the test class's real qualified name — that's expected and proves the
        allowlist gate, not a downstream check, is what changed.)
        """
        data = _registered_untrusted_component["data"]
        # Without allowed_modules, this is rejected as untrusted.
        with pytest.raises(DeserializationError, match="not on the trusted-module allowlist"):
            Pipeline.from_dict(data)
        # With the matching pattern, the allowlist gate passes; the failure now comes from
        # the qualified-name mismatch in default_from_dict — a downstream check.
        with pytest.raises(DeserializationError, match="can't be deserialized as"):
            Pipeline.from_dict(data, allowed_modules=["evilpkg.*"])


class TestPipelineLoadAndLoadsPropagation:
    """
    Verify that the security kwargs added to `Pipeline.from_dict` are propagated correctly
    through the `Pipeline.loads` (string) and `Pipeline.load` (file-like) entry points, and that
    they produce equivalent behavior to calling `from_dict` directly.
    """

    @staticmethod
    def _yaml_for(data: dict) -> str:
        # We can't round-trip through `Pipeline.from_dict` + `dumps` because the registered
        # `EvilComponent`'s real qualified name doesn't match the fake type — the inner
        # `default_from_dict` would reject it. Build the YAML directly via the marshaller instead.
        return YamlMarshaller().marshal(data)

    def test_loads_rejects_untrusted_by_default(self, _registered_untrusted_component):
        yaml_str = self._yaml_for(_registered_untrusted_component["data"])
        with pytest.raises(DeserializationError, match="not on the trusted-module allowlist"):
            Pipeline.loads(yaml_str)

    def test_loads_propagates_allowed_modules(self, _registered_untrusted_component):
        yaml_str = self._yaml_for(_registered_untrusted_component["data"])
        # With the matching pattern, the allowlist gate passes; downstream we get the type
        # mismatch — proving the kwarg reached the gate.
        with pytest.raises(DeserializationError, match="can't be deserialized as"):
            Pipeline.loads(yaml_str, allowed_modules=["evilpkg.*"])

    def test_loads_propagates_unsafe(self, _registered_untrusted_component):
        yaml_str = self._yaml_for(_registered_untrusted_component["data"])
        # `unsafe=True` bypasses the allowlist entirely; downstream we still get the type mismatch.
        with pytest.raises(DeserializationError, match="can't be deserialized as"):
            Pipeline.loads(yaml_str, unsafe=True)

    def test_load_rejects_untrusted_by_default(self, _registered_untrusted_component):
        yaml_str = self._yaml_for(_registered_untrusted_component["data"])
        with pytest.raises(DeserializationError, match="not on the trusted-module allowlist"):
            Pipeline.load(io.StringIO(yaml_str))

    def test_load_propagates_allowed_modules(self, _registered_untrusted_component):
        yaml_str = self._yaml_for(_registered_untrusted_component["data"])
        with pytest.raises(DeserializationError, match="can't be deserialized as"):
            Pipeline.load(io.StringIO(yaml_str), allowed_modules=["evilpkg.*"])

    def test_load_propagates_unsafe(self, _registered_untrusted_component):
        yaml_str = self._yaml_for(_registered_untrusted_component["data"])
        with pytest.raises(DeserializationError, match="can't be deserialized as"):
            Pipeline.load(io.StringIO(yaml_str), unsafe=True)

    def test_load_loads_from_dict_equivalent_on_rejection(self, _registered_untrusted_component):
        """All three entry points produce the same rejection message for the same untrusted payload."""
        data = _registered_untrusted_component["data"]
        yaml_str = self._yaml_for(data)

        def _capture(callable_: Callable[[], object]) -> str:
            with pytest.raises(DeserializationError) as exc_info:
                callable_()
            return str(exc_info.value)

        from_dict_msg = _capture(lambda: Pipeline.from_dict(data))
        loads_msg = _capture(lambda: Pipeline.loads(yaml_str))
        load_msg = _capture(lambda: Pipeline.load(io.StringIO(yaml_str)))

        assert "not on the trusted-module allowlist" in from_dict_msg
        assert from_dict_msg == loads_msg == load_msg
