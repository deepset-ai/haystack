# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack.utils import HAYSTACK_PROGRESS_ENV, get_progress_bar_setting


def test_progress_bar_env_disabled(monkeypatch):
    """HAYSTACK_PROGRESS_BARS set to disable values should disable progress bars."""
    disable_values = ["0", "false", "no", "off", "disable", "disabled"]
    for val in disable_values:
        monkeypatch.setenv(HAYSTACK_PROGRESS_ENV, val)
        assert get_progress_bar_setting(component_default=True) is False


def test_progress_bar_env_enabled(monkeypatch):
    """HAYSTACK_PROGRESS_BARS set to enable values should enable progress bars."""
    enable_values = ["1", "true", "yes", "on", "enable", "enabled"]
    for val in enable_values:
        monkeypatch.setenv(HAYSTACK_PROGRESS_ENV, val)
        assert get_progress_bar_setting(component_default=False) is True


def test_progress_bar_env_case_insensitive(monkeypatch):
    """Environment variable should be case-insensitive."""
    monkeypatch.setenv(HAYSTACK_PROGRESS_ENV, "TRUE")
    assert get_progress_bar_setting(component_default=False) is True

    monkeypatch.setenv(HAYSTACK_PROGRESS_ENV, "FALSE")
    assert get_progress_bar_setting(component_default=True) is False

    monkeypatch.setenv(HAYSTACK_PROGRESS_ENV, "OFF")
    assert get_progress_bar_setting(component_default=True) is False


def test_progress_bar_env_unset(monkeypatch):
    """Unset environment variable should use component default."""
    monkeypatch.delenv(HAYSTACK_PROGRESS_ENV, raising=False)

    assert get_progress_bar_setting(component_default=True) is True
    assert get_progress_bar_setting(component_default=False) is False


def test_progress_bar_env_invalid_value(monkeypatch):
    """Invalid environment variable value should fall back to component default."""
    invalid_values = ["", "invalid", "2", "maybe", "2.5"]
    for val in invalid_values:
        monkeypatch.setenv(HAYSTACK_PROGRESS_ENV, val)
        # Should fall back to component_default
        assert get_progress_bar_setting(component_default=True) is True
        assert get_progress_bar_setting(component_default=False) is False


def test_progress_bar_custom_env_var(monkeypatch):
    """Should support custom environment variable name."""
    monkeypatch.setenv("MY_CUSTOM_PROGRESS_VAR", "0")
    assert get_progress_bar_setting(component_default=True, env_var="MY_CUSTOM_PROGRESS_VAR") is False

    monkeypatch.setenv("MY_CUSTOM_PROGRESS_VAR", "1")
    assert get_progress_bar_setting(component_default=False, env_var="MY_CUSTOM_PROGRESS_VAR") is True


def test_progress_bar_constant_accessible():
    """HAYSTACK_PROGRESS_ENV constant should be accessible."""
    assert HAYSTACK_PROGRESS_ENV == "HAYSTACK_PROGRESS_BARS"


def test_progress_bar_default_parameter():
    """Component default should be used when env var is not set."""
    # No env var set, should return component default
    assert get_progress_bar_setting() is True
    assert get_progress_bar_setting(component_default=False) is False
