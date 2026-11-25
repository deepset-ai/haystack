# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack.utils.progress import _get_progress_bar_setting


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", " TRUE ", "On"])
def test_environment_can_enable_progress_bars(monkeypatch, value):
    monkeypatch.setenv("HAYSTACK_PROGRESS_BARS", value)
    assert _get_progress_bar_setting(default=False) is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", " FALSE ", "Off"])
def test_environment_can_disable_progress_bars(monkeypatch, value):
    monkeypatch.setenv("HAYSTACK_PROGRESS_BARS", value)
    assert _get_progress_bar_setting(default=True) is False


@pytest.mark.parametrize("default", [True, False])
@pytest.mark.parametrize("value", [None, "", "  ", "invalid"])
def test_missing_or_unrecognized_override_preserves_component_setting(monkeypatch, default, value):
    if value is None:
        monkeypatch.delenv("HAYSTACK_PROGRESS_BARS", raising=False)
    else:
        monkeypatch.setenv("HAYSTACK_PROGRESS_BARS", value)
    assert _get_progress_bar_setting(default=default) is default
