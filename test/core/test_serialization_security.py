# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import functools
import io
import json
import logging
import operator
import os
import subprocess
import types
from collections.abc import Callable
from unittest import mock

import pytest

from haystack import component as component_module
from haystack.core import serialization_security as ss
from haystack.core.errors import DeserializationError
from haystack.core.pipeline import Pipeline
from haystack.core.serialization import (
    allow_deserialization_module,
    default_from_dict,
    generate_qualified_class_name,
    import_class_by_name,
)
from haystack.core.serialization_security import (
    _DENIED_BUILTIN_NAMES,
    DESERIALIZATION_ALLOWLIST_ENV_VAR,
    UNSAFE_DESERIALIZATION_ENV_VAR,
    _check_module_allowed,
    _current_context,
    _deserialization_context,
    _DeserializationContext,
    _extra_allowed_modules,
    _is_module_allowed,
    _is_unsafe_deserialization,
    _module_matches,
)
from haystack.marshal import YamlMarshaller
from haystack.utils import deserialize_callable, type_serialization
from haystack.utils.type_serialization import deserialize_type


@pytest.fixture(autouse=True)
def _reset_allowlist_state(monkeypatch):
    """
    Force a clean (safe-default, no extra patterns) state for every test in this module so we are
    testing the actual security model. The top-level test conftest extends the process-wide
    allowlist with test-only patterns (`test_*`, `pydantic`, ...); we must clear those here so
    "untrusted" really means untrusted.
    """
    monkeypatch.delenv(DESERIALIZATION_ALLOWLIST_ENV_VAR, raising=False)
    monkeypatch.delenv(UNSAFE_DESERIALIZATION_ENV_VAR, raising=False)
    # Clear the frozen env-var snapshot so each test starts like a fresh process. No `raising=False`
    # here on purpose: if the attribute is ever renamed, this must fail loudly rather than silently
    # become a no-op and leak one test's mode into the next.
    monkeypatch.setattr(ss, "_unsafe_env_snapshot", None)
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


class TestUnsafeDeserializationEnvVar:
    """
    `HAYSTACK_UNSAFE_DESERIALIZATION` is a process-wide off switch: when truthy it makes every load
    behave as if `unsafe=True` was passed, disabling all deserialization safety checks. It is a
    separate axis from the module allowlist (`HAYSTACK_DESERIALIZATION_ALLOWLIST`), which only widens
    which modules may be imported.
    """

    def test_unset_keeps_safe_mode(self):
        # Sanity: with the env var unset, the guards are active.
        assert not _is_module_allowed("subprocess")
        with pytest.raises(DeserializationError):
            deserialize_callable("os.system")

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "True"])
    def test_truthy_values_disable_all_checks(self, monkeypatch, value):
        monkeypatch.setenv(UNSAFE_DESERIALIZATION_ENV_VAR, value)
        # Allowlist bypassed ...
        assert _is_module_allowed("subprocess")
        assert deserialize_callable("os.system") is __import__("os").system
        # ... and so are the denylists / control-plane / traversal guards.
        assert callable(deserialize_callable("builtins.eval"))
        assert callable(deserialize_callable("haystack.core.serialization_security.allow_deserialization_module"))
        assert callable(
            deserialize_callable("haystack.core.serialization_security.allow_deserialization_module.__globals__.get")
        )

    @pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "yes", "on", "nope"])
    def test_falsey_values_keep_safe_mode(self, monkeypatch, value):
        monkeypatch.setenv(UNSAFE_DESERIALIZATION_ENV_VAR, value)
        with pytest.raises(DeserializationError):
            deserialize_callable("os.system")

    def test_allows_unsafe_output_adapter_component(self, monkeypatch):
        # A serialized OutputAdapter with `unsafe: true` loads under the env var, exactly as it would
        # under `Pipeline.loads(..., unsafe=True)`.
        yaml = (
            "components:\n"
            "  adapter:\n"
            "    type: haystack.components.converters.output_adapter.OutputAdapter\n"
            "    init_parameters:\n"
            '      template: "{{ documents[0] }}"\n'
            "      output_type: str\n"
            "      unsafe: true\n"
            "connections: []\n"
        )
        with pytest.raises(DeserializationError):
            Pipeline.loads(yaml)  # refused in safe mode
        # The safe-mode load above froze the snapshot, so simulate a fresh process before enabling
        # the env var — a mid-process flip is ignored by design (see the freezing tests below).
        monkeypatch.setenv(UNSAFE_DESERIALIZATION_ENV_VAR, "1")
        monkeypatch.setattr(ss, "_unsafe_env_snapshot", None)
        Pipeline.loads(yaml)  # accepted with the env var set

    def test_warns_once_when_active(self, monkeypatch, caplog):
        monkeypatch.setenv(UNSAFE_DESERIALIZATION_ENV_VAR, "1")
        with caplog.at_level(logging.WARNING):
            assert ss._unsafe_env_enabled() is True
            assert ss._unsafe_env_enabled() is True
        warnings = [r for r in caplog.records if UNSAFE_DESERIALIZATION_ENV_VAR in r.getMessage()]
        assert len(warnings) == 1, "the safety-disabled warning must be logged exactly once per process"
        assert "DISABLED" in warnings[0].getMessage()

    def test_value_is_frozen_after_the_first_check(self, monkeypatch):
        # The first check (here: a safe-mode resolution) snapshots the env var ...
        with pytest.raises(DeserializationError):
            deserialize_callable("os.system")
        # ... so setting it afterwards has no effect for the rest of the process.
        monkeypatch.setenv(UNSAFE_DESERIALIZATION_ENV_VAR, "1")
        assert not _is_module_allowed("subprocess")
        with pytest.raises(DeserializationError):
            deserialize_callable("os.system")

    def test_freezing_applies_in_both_directions(self, monkeypatch):
        # Symmetrically, unsetting it after the snapshot does not re-arm the checks: the switch is
        # decided once, so the mode of a process cannot change under a caller's feet mid-run.
        monkeypatch.setenv(UNSAFE_DESERIALIZATION_ENV_VAR, "1")
        assert _is_module_allowed("subprocess")
        monkeypatch.delenv(UNSAFE_DESERIALIZATION_ENV_VAR)
        assert _is_module_allowed("subprocess")

    def test_env_write_gadget_is_not_resolvable_from_serialized_data(self, monkeypatch):
        """
        First line of defense: the gadget cannot be resolved at all.

        `os.environ.update` is `collections.abc.MutableMapping.update`, and `collections` is on the
        default allowlist — so a handle that walks an allowlisted module's scope-level `os.environ`
        binding ends at a callable whose own module *is* allowed. The per-hop check on the real
        module of every traversed object rejects the `environ` hop itself, because that object
        belongs to the un-allowlisted `os`.
        """
        monkeypatch.setattr(type_serialization, "environ", os.environ, raising=False)
        with pytest.raises(DeserializationError, match="module 'os'"):
            deserialize_callable("haystack.utils.type_serialization.environ.update")

    def test_env_write_reachable_from_serialized_data_cannot_disable_safety(self, monkeypatch):
        """
        Second line of defense, and the reason the snapshot is frozen.

        Even granting the attacker the env write that the gadget above would have performed — say
        via a mutator the traversal check cannot see, or as an `OutputAdapter` Jinja
        `custom_filters` entry that runs while the component is being constructed — the ongoing
        load must not flip into unsafe mode. Were the env var read fresh on every check, it would,
        handing the pipeline arbitrary code execution.
        """
        # Any deserialization check snapshots the env var, so by the time the attacker's payload
        # runs the mode is already frozen — as it would be mid-load in a real process. Assert the
        # safe-mode baseline here to take that snapshot without relying on the gadget above.
        assert not _is_module_allowed("subprocess")

        # Register the variable before the write: the write below goes straight to the real
        # environment, so monkeypatch has to know the pre-attack state to undo it at teardown.
        # (Registering afterwards would record the polluted value and leak it into other tests.)
        monkeypatch.setenv(UNSAFE_DESERIALIZATION_ENV_VAR, "")
        os.environ.update({UNSAFE_DESERIALIZATION_ENV_VAR: "1"})
        assert os.environ[UNSAFE_DESERIALIZATION_ENV_VAR] == "1"  # the write lands ...

        assert not _is_unsafe_deserialization()  # ... and changes nothing
        assert not _is_module_allowed("subprocess")
        with pytest.raises(DeserializationError):
            deserialize_callable("os.system")
        with pytest.raises(DeserializationError):
            deserialize_callable("builtins.eval")


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

    # `type` is excluded: it is a valid class in this path (covered by test_allows_builtin_type),
    # even though it is denied as a *callable*. Every other denied builtin is a function, not a type.
    @pytest.mark.parametrize("name", sorted(_DENIED_BUILTIN_NAMES - {"type"}))
    def test_rejects_dangerous_builtin(self, name):
        # `builtins` passes the module allowlist, so a class reference must resolve to an actual
        # type — otherwise a nested `{"type": "builtins.compile"}` payload would resolve a function.
        # The dunder-named builtins (`__import__`, `__build_class__`) are refused even earlier by the
        # object-internals traversal guard.
        with pytest.raises(DeserializationError, match="not a type|internal attribute"):
            import_class_by_name(f"builtins.{name}")

    def test_allows_builtin_type(self):
        # Builtin types must still import as classes. `type` is a valid class here even though it
        # is blocked as a *callable* by deserialize_callable.
        assert import_class_by_name("builtins.dict") is dict
        assert import_class_by_name("builtins.type") is type


class TestDefaultFromDictNestedBuiltins:
    def test_nested_dangerous_builtin_rejected(self):
        """A nested `{"type": "builtins.<dangerous>"}` payload must be rejected via the same gate."""

        class Container:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        data = {
            "type": generate_qualified_class_name(Container),
            "init_parameters": {"payload": {"type": "builtins.compile", "init_parameters": {}}},
        }
        with pytest.raises(DeserializationError, match="not a type"):
            default_from_dict(Container, data)


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
        # No prefix of `subprocess.Popen` matches the default allowlist (or `unrelated`). This also
        # covers the plain default-allowlist case, since the context only ever appends patterns.
        with _deserialization_context(allowed_modules=["unrelated"]):
            with pytest.raises(DeserializationError, match="not on the trusted-module allowlist"):
                deserialize_callable("subprocess.Popen")


class TestDeniedBuiltins:
    """
    `builtins` is on the default allowlist so harmless members (`builtins.print`, builtin types)
    round-trip, but the module-granular allowlist is too coarse to stop dangerous builtin callables
    like `eval`/`exec`. Those are blocked in the callable-resolution path regardless of the
    allowlist (parametrized over the canonical denied set so additions are covered automatically).
    """

    @pytest.mark.parametrize("name", _DENIED_BUILTIN_NAMES)
    def test_dangerous_builtin_callable_rejected(self, name):
        # Most denied builtins are refused by the identity denylist ("blocked because it can be
        # used ..."); the dunder-named ones (`__import__`, `__build_class__`) are caught earlier by
        # the object-internals traversal guard. Either way they never resolve.
        with pytest.raises(DeserializationError, match="blocked because it can be used|internal attribute"):
            deserialize_callable(f"builtins.{name}")

    def test_harmless_builtin_callable_still_resolves(self):
        # `serialize_callable(print)` emits "builtins.print"; harmless builtins must keep
        # round-tripping (only the denied set is blocked).
        assert deserialize_callable("builtins.print") is print
        assert deserialize_callable("builtins.len") is len
        assert deserialize_callable("builtins.sorted") is sorted

    def test_alias_to_dangerous_builtin_rejected(self):
        # `io.open is builtins.open`, so the identity-based check catches it once `io` is allowed.
        with _deserialization_context(allowed_modules=["io"]):
            with pytest.raises(DeserializationError, match="blocked because it can be used"):
                deserialize_callable("io.open")

    def test_type_blocked_as_callable_but_allowed_as_type(self):
        # `type` is a class-creation gadget as a callable, but a legitimate type annotation.
        with pytest.raises(DeserializationError, match="blocked because it can be used"):
            deserialize_callable("builtins.type")
        assert deserialize_type("type") is type

    def test_unsafe_mode_bypasses_the_block(self):
        # `unsafe=True` disables all deserialization safety checks by design.
        with _deserialization_context(unsafe=True):
            assert deserialize_callable("builtins.eval") is eval


class TestDeniedImportPrimitives:
    """
    An import primitive that lives inside an allowlisted namespace slips through both the
    module-granular allowlist (its `__module__` is allowlisted) and the builtin denylist (it is not
    a `builtins` member). It is a functional twin of the denied builtin `__import__` — a gateway to
    `os`/`subprocess`/... — and must be blocked.
    """

    def test_thread_safe_import_rejected(self):
        with pytest.raises(DeserializationError, match="import primitive"):
            deserialize_callable("haystack.utils.type_serialization.thread_safe_import")

    def test_unsafe_mode_bypasses_the_block(self):
        from haystack.utils.type_serialization import thread_safe_import

        with _deserialization_context(unsafe=True):
            handle = "haystack.utils.type_serialization.thread_safe_import"
            assert deserialize_callable(handle) is thread_safe_import

    def test_pipeline_loads_rejects_import_filter_gadget(self):
        # End-to-end through the default-safe API: an attacker registers the import primitive as an
        # OutputAdapter custom filter and uses it in the template to import `os` and run a command.
        # Loading must reject it before any code runs.
        gadget_yaml = (
            "components:\n"
            "  adapter:\n"
            "    type: haystack.components.converters.output_adapter.OutputAdapter\n"
            "    init_parameters:\n"
            "      template: \"{{ ('os' | imp).system('echo pwned') }}\"\n"
            "      output_type: str\n"
            "      custom_filters:\n"
            '        imp: "haystack.utils.type_serialization.thread_safe_import"\n'
            "connections: []\n"
        )
        with mock.patch("os.system") as mocked_system:
            with pytest.raises(DeserializationError):
                Pipeline.loads(gadget_yaml)
        assert not mocked_system.called


class TestDeniedDeserializerInternals:
    """
    The allowlist admits the whole `haystack` namespace so Haystack can deserialize its own
    components. That also makes the deserializer's *own* interface resolvable from serialized data:
    the allowlist-administration function `allow_deserialization_module` and the resolution helpers
    (`deserialize_callable`, `deserialize_type`, `import_class_by_name`). A hostile pipeline can register
    `allow_deserialization_module` as a Jinja custom filter, call it with `"*"` to disarm the
    allowlist process-wide, then use the equally-resolvable `deserialize_callable` to resolve and
    invoke `os.system`. The resolver must refuse to hand any of them back.
    """

    INTERNAL_HANDLES = [
        "haystack.core.serialization_security.allow_deserialization_module",
        "haystack.utils.callable_serialization.deserialize_callable",
        "haystack.utils.type_serialization.deserialize_type",
        "haystack.utils.type_serialization._import_class_by_name",
        "haystack.core.serialization.import_class_by_name",
    ]

    @pytest.mark.parametrize("handle", INTERNAL_HANDLES)
    def test_deserialize_callable_rejects_internal(self, handle):
        with pytest.raises(DeserializationError, match="deserialization control plane"):
            deserialize_callable(handle)

    @pytest.mark.parametrize("handle", INTERNAL_HANDLES)
    def test_unsafe_mode_bypasses_the_block(self, handle):
        # unsafe=True disables all deserialization safety checks by design.
        with _deserialization_context(unsafe=True):
            assert callable(deserialize_callable(handle))

    def test_all_internal_functions_carry_the_marker(self):
        # By construction: every internal entry point is stamped, so a new resolver helper is
        # excluded the day it is written (once decorated), not the day it is exploited.
        from haystack.core.serialization import import_class_by_name as _icbn
        from haystack.core.serialization_security import _DESERIALIZATION_INTERNAL_ATTR
        from haystack.core.serialization_security import allow_deserialization_module as _adm
        from haystack.utils.callable_serialization import deserialize_callable as _dc
        from haystack.utils.type_serialization import _import_class_by_name as _icbn_impl
        from haystack.utils.type_serialization import deserialize_type as _dt

        for func in (_adm, _dc, _dt, _icbn, _icbn_impl):
            assert getattr(func, _DESERIALIZATION_INTERNAL_ATTR, False) is True

    def test_class_resolution_path_rejects_internal(self):
        # The admin/resolution API must also be unreachable as a component `type` or class reference.
        handle = "haystack.core.serialization_security.allow_deserialization_module"
        with pytest.raises((DeserializationError, ImportError)):
            import_class_by_name(handle)
        with pytest.raises((DeserializationError, ImportError)):
            deserialize_type(handle)

    def _self_disable_yaml(self, component_type, extra_init):
        filters = (
            "        allow: haystack.core.serialization_security.allow_deserialization_module\n"
            "        dc: haystack.utils.callable_serialization.deserialize_callable\n"
            "        mp: builtins.map\n"
        )
        return (
            "components:\n"
            "  c:\n"
            f"    type: {component_type}\n"
            "    init_parameters:\n"
            f"{extra_init}"
            "      custom_filters:\n"
            f"{filters}"
            "      unsafe: false\n"
            "connections: []\n"
        )

    def test_pipeline_loads_rejects_output_adapter_self_disable_vector(self):
        # End-to-end reproduction of the reported RCE via OutputAdapter, in default safe mode. The
        # broad serialized-filter gate rejects it before any control-plane callable is resolved.
        yaml = self._self_disable_yaml(
            "haystack.components.converters.output_adapter.OutputAdapter",
            "      template: \"{{ '*' | allow }}{{ 'os.system' | dc | mp(cmds) | list }}\"\n      output_type: str\n",
        )
        with mock.patch("os.system") as mocked_system:
            with pytest.raises(DeserializationError, match="custom filters while loading in safe mode"):
                Pipeline.loads(yaml)
        assert not mocked_system.called

    def test_pipeline_loads_rejects_conditional_router_self_disable_vector(self):
        # Same chain carried by ConditionalRouter; its serialized filters are rejected before the
        # narrower control-plane callable checks need to run.
        extra = (
            "      routes:\n"
            "        - condition: \"{{ ['*'|allow, ('os.system'|dc|mp(cmds)|list)] and True }}\"\n"
            '          output: "{{ 1 }}"\n'
            "          output_name: out\n"
            "          output_type: int\n"
        )
        yaml = self._self_disable_yaml("haystack.components.routers.conditional_router.ConditionalRouter", extra)
        with mock.patch("os.system") as mocked_system:
            with pytest.raises(DeserializationError, match="custom filters while loading in safe mode"):
                Pipeline.loads(yaml)
        assert not mocked_system.called

    def test_rejected_load_does_not_widen_the_process_wide_allowlist(self):
        # The chain is cut before `allow_deserialization_module` can run, so the second reported
        # consequence — permanent process-wide teardown of the allowlist — never happens.
        yaml = self._self_disable_yaml(
            "haystack.components.converters.output_adapter.OutputAdapter",
            "      template: \"{{ '*' | allow }}{{ 'os.system' | dc | mp(cmds) | list }}\"\n      output_type: str\n",
        )
        with pytest.raises(DeserializationError):
            Pipeline.loads(yaml)
        assert _extra_allowed_modules == []
        with pytest.raises(DeserializationError):
            deserialize_callable("os.system")

    # The mutable control-plane state (the allowlist list and the deserialization context var) is
    # reachable through the same attribute walk the resolver performs, so a bound mutator can widen
    # the allowlist without touching any of the blocked resolver functions above.
    CONTROL_PLANE_HANDLES = [
        "haystack.core.serialization_security._extra_allowed_modules.append",
        "haystack.core.serialization_security._extra_allowed_modules.extend",
        "haystack.core.serialization_security._extra_allowed_modules.insert",
        "haystack.core.serialization_security._current_context.set",
        "haystack.core.serialization_security._DeserializationContext",
        "haystack.core.serialization_security._get_context",
    ]

    @pytest.mark.parametrize("handle", CONTROL_PLANE_HANDLES)
    def test_deserialize_callable_rejects_control_plane_state(self, handle):
        with pytest.raises(DeserializationError, match="deserialization control plane"):
            deserialize_callable(handle)

    def test_reexported_alias_handles_are_also_rejected(self):
        # The same function objects are re-exported under other public paths; they inherit the mark
        # (or match by defining module) and must be refused there too.
        for handle in (
            "haystack.core.serialization.allow_deserialization_module",
            "haystack.utils.deserialize_callable",
            "haystack.utils.deserialize_type",
        ):
            with pytest.raises(DeserializationError, match="deserialization control plane"):
                deserialize_callable(handle)

    def test_pipeline_loads_rejects_allowlist_poisoning_via_append(self):
        # A staged attack that never touches the blocked resolver helpers: a custom filter bound to
        # `_extra_allowed_modules.append`, called with "*" from the template, would disarm the
        # allowlist process-wide (the advisory's persistent "second consequence") and enable a staged
        # RCE on a *later* load that resolves `os.system` directly. Blocking it at load time prevents
        # both the poisoning and the later resolution.
        poison_yaml = (
            "components:\n"
            "  adapter:\n"
            "    type: haystack.components.converters.output_adapter.OutputAdapter\n"
            "    init_parameters:\n"
            "      template: \"{{ '*' | ap }}{{ trigger }}\"\n"
            "      output_type: str\n"
            "      custom_filters:\n"
            "        ap: haystack.core.serialization_security._extra_allowed_modules.append\n"
            "      unsafe: false\n"
            "connections: []\n"
        )
        with pytest.raises(DeserializationError, match="custom filters while loading in safe mode"):
            Pipeline.loads(poison_yaml)
        assert _extra_allowed_modules == []
        with pytest.raises(DeserializationError):
            deserialize_callable("os.system")


class TestDeniedPipelineEntryPoints:
    """
    Blocking the low-level resolvers (`deserialize_callable` etc.) is not enough on its own: the
    high-level loading entry points that accept `unsafe=True` — `Pipeline.loads` / `load` /
    `from_dict` — and the execute primitives (`Pipeline.run` / `run_async` / `run_async_generator` /
    `stream`) are themselves resolvable
    from the allowlisted `haystack` namespace. Bound as sandbox-bypassing `custom_filters`, they
    re-enter deserialization: a safe-mode pipeline can call `Pipeline.loads(nested, unsafe=True)` to
    load a *nested* pipeline whose own filters (`allow_deserialization_module`, `deserialize_callable`)
    bind under the nested unsafe context, then `Pipeline.run` it to poison the process-wide allowlist
    with `"*"` and invoke `os.system`. These entry points must be part of the deserialization control
    plane too, so they can never be produced by deserializing untrusted data.
    """

    ENTRY_POINTS = [
        "haystack.core.pipeline.pipeline.Pipeline.loads",
        "haystack.core.pipeline.pipeline.Pipeline.load",
        "haystack.core.pipeline.pipeline.Pipeline.from_dict",
        "haystack.core.pipeline.pipeline.Pipeline.run",
        "haystack.core.pipeline.pipeline.Pipeline.run_async",
        # `run_async_generator` and `stream` reach `run_async` too, so they are execute primitives
        # in their own right: `stream` schedules the run via `asyncio.create_task`.
        "haystack.core.pipeline.pipeline.Pipeline.run_async_generator",
        "haystack.core.pipeline.pipeline.Pipeline.stream",
        # The base class the classmethods actually live on, and the public re-export.
        "haystack.core.pipeline.base.PipelineBase.loads",
        "haystack.Pipeline.loads",
    ]

    @pytest.mark.parametrize("handle", ENTRY_POINTS)
    def test_deserialize_callable_rejects_pipeline_entry_points(self, handle):
        with pytest.raises(DeserializationError, match="deserialization control plane"):
            deserialize_callable(handle)

    @pytest.mark.parametrize("handle", ENTRY_POINTS)
    def test_unsafe_mode_bypasses_the_block(self, handle):
        # unsafe=True disables all deserialization safety checks by design.
        with _deserialization_context(unsafe=True):
            assert callable(deserialize_callable(handle))

    def test_entry_points_carry_the_marker(self):
        # By construction: the loaders and the execute primitives are stamped, so the resolver
        # excludes them the day they are marked, not the day the chain below is discovered.
        from haystack.core.pipeline.base import PipelineBase
        from haystack.core.pipeline.pipeline import Pipeline as ConcretePipeline
        from haystack.core.serialization_security import _DESERIALIZATION_INTERNAL_ATTR

        for classmethod_name in ("loads", "load", "from_dict"):
            func = getattr(PipelineBase, classmethod_name).__func__
            assert getattr(func, _DESERIALIZATION_INTERNAL_ATTR, False) is True
        for method_name in ("run", "run_async", "run_async_generator", "stream"):
            func = getattr(ConcretePipeline, method_name)
            assert getattr(func, _DESERIALIZATION_INTERNAL_ATTR, False) is True

    def _nested_yaml(self):
        # Meant to be loaded with unsafe=True (so its `allow`/`dc` filters bind), then run: it poisons
        # the process-wide allowlist with "*" and resolves and invokes os.system — the original gadget.
        return (
            "components:\n"
            "  c:\n"
            "    type: haystack.components.converters.output_adapter.OutputAdapter\n"
            "    init_parameters:\n"
            "      template: \"{{ ('*' | allow) }}{{ ('os.system' | dc | mp([cmd]) | list) }}\"\n"
            "      output_type: str\n"
            "      custom_filters:\n"
            "        allow: haystack.core.serialization_security.allow_deserialization_module\n"
            "        dc: haystack.utils.callable_serialization.deserialize_callable\n"
            "        mp: builtins.map\n"
            "      unsafe: true\n"
            "connections: []\n"
        )

    def test_pipeline_loads_rejects_nested_unsafe_load_run_vector(self):
        # End-to-end reproduction, in default safe mode. The top pipeline binds Pipeline.loads (L) and
        # Pipeline.run (R) as custom_filters; the broad serialized-filter gate fires before resolving
        # L, so os.system is never reached and the allowlist is never poisoned.
        top_yaml = (
            "components:\n"
            "  c:\n"
            "    type: haystack.components.converters.output_adapter.OutputAdapter\n"
            "    init_parameters:\n"
            "      template: \"{{ nested | L(unsafe=True) | R(data={'c': {'cmd': cmd}}) }}\"\n"
            "      output_type: str\n"
            "      custom_filters:\n"
            "        L: haystack.core.pipeline.pipeline.Pipeline.loads\n"
            "        R: haystack.core.pipeline.pipeline.Pipeline.run\n"
            "      unsafe: false\n"
            "connections: []\n"
        )
        with mock.patch("os.system") as mocked_system:
            with pytest.raises(DeserializationError, match="custom filters while loading in safe mode"):
                Pipeline.loads(top_yaml)
        assert not mocked_system.called
        assert _extra_allowed_modules == []
        with pytest.raises(DeserializationError):
            deserialize_callable("os.system")

    def test_nested_gadget_pipeline_needs_an_unsafe_load_to_arm(self):
        # The other half of the chain, on its own: the nested payload is inert unless something loads
        # it in unsafe mode. In safe mode the load is refused (its `unsafe: true` OutputAdapter, and
        # its control-plane filters, are both rejected), so the gadget never arms.
        with mock.patch("os.system") as mocked_system:
            with pytest.raises(DeserializationError, match="unsafe=True while loading in safe mode"):
                Pipeline.loads(self._nested_yaml())
        assert not mocked_system.called
        assert _extra_allowed_modules == []

        # Under an unsafe load it arms *and fires* — and it fires at load time, not at run time:
        # `OutputAdapter.__init__` extracts template variables via `meta.find_undeclared_variables`,
        # whose codegen constant-folds filter calls with constant arguments, so `('*' | allow)` runs
        # while the component is being constructed. No `.run()` is involved.
        # This is why blocking the *loaders* is the load-bearing part of the fix: once safe-mode data
        # cannot reach an unsafe load, it cannot reach this construction either.
        Pipeline.loads(self._nested_yaml(), unsafe=True)
        assert _extra_allowed_modules == ["*"]  # reverted by the autouse fixture


class TestPipelineCallableClassification:
    """
    Structural guard for the marking above. `mark_deserialization_internal` is a per-callable
    denylist inside a namespace-wide allowlist, so its completeness depends on someone remembering
    to stamp the next dangerous entry point — exactly how `run_async_generator` and `stream` were
    missed when `run` / `run_async` were marked.

    Rather than testing the callables we happen to have thought of, this enumerates every public
    callable on `PipelineBase` / `Pipeline` and requires each one to be classified here. Adding a
    new public method fails this test until it is declared either deserializer-internal or safe to
    resolve, and dropping a decorator fails it too.
    """

    # Loaders that accept `unsafe=True`, plus the execute primitives.
    INTERNAL: dict[str, set[str]] = {
        "PipelineBase": {"from_dict", "load", "loads"},
        "Pipeline": {"run", "run_async", "run_async_generator", "stream"},
    }
    # Resolvable from serialized data without handing it deserialization control or execution:
    # graph construction, introspection, serialization *out*, and lifecycle.
    SAFE_TO_RESOLVE: dict[str, set[str]] = {
        "PipelineBase": {
            "add_component",
            "add_components",
            "close",
            "close_async",
            "connect",
            "connect_many",
            "draw",
            "dump",
            "dumps",
            "get_component",
            "get_component_name",
            "inputs",
            "outputs",
            "remove_component",
            "show",
            "to_dict",
            "validate_input",
            "validate_pipeline",
            "walk",
            "warm_up",
            "warm_up_async",
        },
        "Pipeline": set(),
    }

    @staticmethod
    def _public_callables(pipeline_class):
        # `vars()` rather than `dir()`: each class is checked against its own declarations, so the
        # two classification sets stay disjoint and a method moving between them is visible.
        return {
            name
            for name, value in vars(pipeline_class).items()
            if not name.startswith("_") and isinstance(value, (types.FunctionType, classmethod, staticmethod))
        }

    @pytest.mark.parametrize("class_name", ["PipelineBase", "Pipeline"])
    def test_every_public_callable_is_classified(self, class_name):
        from haystack.core.pipeline.base import PipelineBase
        from haystack.core.pipeline.pipeline import Pipeline as ConcretePipeline
        from haystack.core.serialization_security import _DESERIALIZATION_INTERNAL_ATTR

        cls = {"PipelineBase": PipelineBase, "Pipeline": ConcretePipeline}[class_name]
        internal, safe = self.INTERNAL[class_name], self.SAFE_TO_RESOLVE[class_name]
        assert not internal & safe, f"{class_name}: classified both ways: {sorted(internal & safe)}"

        public = self._public_callables(cls)
        unclassified = public - internal - safe
        assert not unclassified, (
            f"{class_name}.{sorted(unclassified)} is public and unclassified. Decide whether it hands "
            f"deserialized data deserialization control or execution: if so, stamp it with "
            f"`mark_deserialization_internal` and add it to INTERNAL; otherwise add it to SAFE_TO_RESOLVE."
        )
        stale = (internal | safe) - public
        assert not stale, f"{class_name}: classified but no longer public: {sorted(stale)}"

        for name in sorted(public):
            attr = getattr(cls, name)
            func = getattr(attr, "__func__", attr)
            marked = getattr(func, _DESERIALIZATION_INTERNAL_ATTR, False) is True
            assert marked is (name in internal), (
                f"{class_name}.{name} is classified as {'internal' if name in internal else 'safe'} "
                f"but its `mark_deserialization_internal` stamp is {marked}."
            )


class TestObjectInternalsTraversal:
    """
    A serialized handle legitimately walks `module.Class.method`, never into an object's internals.
    Descending into dunder attributes (`__globals__`, `__dict__`, `__class__`, `__subclasses__`, ...)
    or the frame/code accessors is a classic sandbox escape: `<func>.__globals__` yields the defining
    module's live namespace, from which the allowlist state can be rewritten or `__builtins__` (hence
    `eval`/`exec`) reached — all while the traversal stays inside an allowlisted module, so neither the
    module allowlist nor the per-object identity checks would catch it.
    """

    TRAVERSAL_GADGETS = [
        # `<func>.__globals__` -> the defining module's live globals dict, then a bound mutator/getter.
        "haystack.core.serialization_security.allow_deserialization_module.__globals__.update",
        "haystack.core.serialization_security.allow_deserialization_module.__globals__.get",
        "haystack.utils.callable_serialization.deserialize_callable.__globals__.get",
        # `<module>.__dict__` -> the module's live namespace dict.
        "haystack.core.serialization_security.__dict__.update",
        # Function/class internals reachable as a final or intermediate hop.
        "haystack.components.converters.output_adapter.OutputAdapter.__init__.__globals__.get",
        "haystack.components.converters.output_adapter.OutputAdapter.__subclasses__",
        "haystack.components.converters.output_adapter.OutputAdapter.__class__",
    ]

    @pytest.mark.parametrize("handle", TRAVERSAL_GADGETS)
    def test_deserialize_callable_rejects_internal_attribute_traversal(self, handle):
        with pytest.raises(DeserializationError, match="internal attribute"):
            deserialize_callable(handle)

    def test_import_class_by_name_rejects_internal_attribute(self):
        with pytest.raises((DeserializationError, ImportError)):
            import_class_by_name("haystack.core.serialization_security.__dict__")

    def test_unsafe_mode_bypasses_the_block(self):
        with _deserialization_context(unsafe=True):
            resolved = deserialize_callable(
                "haystack.core.serialization_security.allow_deserialization_module.__globals__.get"
            )
            assert callable(resolved)

    def test_legitimate_attribute_walks_still_resolve(self):
        # `Class.method` and nested traversal through non-dunder attributes must keep working.
        assert callable(deserialize_callable("collections.OrderedDict.fromkeys"))
        assert callable(deserialize_callable("collections.Counter.most_common"))

    def test_pipeline_loads_rejects_globals_rewrite_vector(self):
        # End-to-end: rewriting `_extra_allowed_modules` via `__globals__.update` in default safe
        # mode. The serialized-filter gate rejects the handle before attribute traversal begins.
        poison_yaml = (
            "components:\n"
            "  adapter:\n"
            "    type: haystack.components.converters.output_adapter.OutputAdapter\n"
            "    init_parameters:\n"
            '      template: "{{ mapping | upd }}{{ trigger }}"\n'
            "      output_type: str\n"
            "      custom_filters:\n"
            "        upd: haystack.core.serialization_security.allow_deserialization_module.__globals__.update\n"
            "      unsafe: false\n"
            "connections: []\n"
        )
        with pytest.raises(DeserializationError, match="custom filters while loading in safe mode"):
            Pipeline.loads(poison_yaml)
        assert _extra_allowed_modules == []
        with pytest.raises(DeserializationError):
            deserialize_callable("os.system")


class TestModuleAttributeWalkBypass:
    """
    The allowlist must be enforced against the module a handle
    *actually resolves to*, not against a string prefix of the declared handle.

    An allowlisted package can expose an object from another module as an attribute. A handle can
    then carry an allowlisted `haystack` prefix while its attribute walk escapes through that object
    into an unallowlisted module. Every intermediate object's real module must be checked, not only
    module objects and the final callable.
    """

    # (handle, module the walk escapes into) — real gadgets present in the default install.
    _GADGETS = [
        ("haystack.utils.auth.os.system", "os"),
        ("haystack.utils.auth.os.popen", "os"),
        ("haystack.components.converters.output_adapter.ast.literal_eval", "ast"),
    ]

    @pytest.mark.parametrize("handle, escaped_module", _GADGETS)
    def test_callable_walk_into_unallowlisted_module_rejected(self, handle, escaped_module):
        with pytest.raises(DeserializationError, match=f"module '{escaped_module}'"):
            deserialize_callable(handle)

    def test_class_path_module_leak_rejected(self):
        # The class path resolves `haystack.utils.auth.os` to the `os` module itself; it must not
        # leak an un-allowlisted module as if it were a class.
        with pytest.raises(DeserializationError, match="module 'os'"):
            import_class_by_name("haystack.utils.auth.os")

    def test_type_path_module_leak_rejected(self):
        with pytest.raises(DeserializationError, match="module 'os'"):
            deserialize_type("haystack.utils.auth.os")

    def test_resolved_module_check_covers_plain_attribute(self, monkeypatch):
        # Defense-in-depth: a dangerous callable bound as a plain (non-module) attribute of an
        # allowlisted module is not caught by the module-walk check but must still be rejected by
        # the resolved-`__module__` gate (`subprocess.getoutput.__module__ == "subprocess"`).
        monkeypatch.setattr("haystack.utils.auth.injected_gadget", subprocess.getoutput, raising=False)
        with pytest.raises(DeserializationError, match="module 'subprocess'"):
            deserialize_callable("haystack.utils.auth.injected_gadget")

    def test_callable_walk_through_reexported_unallowlisted_class_rejected(self):
        # `Console` is re-exported from an allowlisted Haystack module but belongs to `rich.console`.
        # Its `_environ` class attribute is the live `os.environ` mapping, whose `update` method
        # reports the default-allowlisted `collections.abc` module. Checking only module hops and the
        # final callable therefore misses the escape; checking the intermediate class rejects it.
        handle = "haystack.hooks.human_in_the_loop.user_interfaces.Console._environ.update"
        with pytest.raises(DeserializationError, match="module 'rich.console'"):
            deserialize_callable(handle)

    def test_legitimate_haystack_callable_still_resolves(self):
        from haystack.utils.callable_serialization import serialize_callable

        handle = "haystack.utils.callable_serialization.serialize_callable"
        assert deserialize_callable(handle) is serialize_callable

    def test_pipeline_loads_rejects_gadget_yaml(self):
        # End-to-end through the public default-safe API: an attacker-supplied pipeline naming the
        # gadget as an `OutputAdapter` custom filter must be rejected at load time.
        gadget_yaml = (
            "components:\n"
            "  adapter:\n"
            "    type: haystack.components.converters.output_adapter.OutputAdapter\n"
            "    init_parameters:\n"
            '      template: "{{ x }}"\n'
            "      output_type: str\n"
            "      custom_filters:\n"
            '        pwn: "haystack.utils.auth.os.system"\n'
            "connections: []\n"
        )
        with pytest.raises(DeserializationError):
            Pipeline.loads(gadget_yaml)


class TestPrivateBackingModuleCompatibility:
    """
    A symbol legitimately exposed by an allowlisted public module can report a *private* C
    accelerator as its `__module__` (`operator.add.__module__ == "_operator"`,
    `io.StringIO.__module__ == "_io"`). The resolved-module gate must still accept these when the
    private module backs the allowlisted module the object was resolved from — otherwise allowing
    a public module would silently fail to resolve its own members and force users to widen trust
    to low-level private modules.
    """

    def test_callable_from_private_backing_module_resolves(self):
        with _deserialization_context(allowed_modules=["operator"]):
            assert deserialize_callable("operator.add") is operator.add

    def test_callable_functools_reduce_resolves(self):
        with _deserialization_context(allowed_modules=["functools"]):
            assert deserialize_callable("functools.reduce") is functools.reduce

    def test_type_from_private_backing_module_resolves(self):
        with _deserialization_context(allowed_modules=["io"]):
            assert deserialize_type("io.StringIO") is io.StringIO

    def test_import_class_from_private_backing_module_resolves(self):
        with _deserialization_context(allowed_modules=["io"]):
            assert import_class_by_name("io.StringIO") is io.StringIO

    def test_exemption_is_scoped_to_the_matching_private_module(self, monkeypatch):
        # The exemption only trusts the private module that backs the *same* allowlisted module.
        # A symbol whose real module is a different (still un-allowlisted) private module must stay
        # blocked: here `operator` is allowed but the injected attribute resolves to `_io`.
        monkeypatch.setattr("operator.injected_gadget", io.StringIO, raising=False)
        with _deserialization_context(allowed_modules=["operator"]):
            with pytest.raises(DeserializationError, match="module '_io'"):
                deserialize_callable("operator.injected_gadget")


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
