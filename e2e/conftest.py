# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from haystack.testing.telemetry import (  # noqa: F401  # autouse fixtures for the whole suite
    block_telemetry_network_calls,
    tag_pipeline_telemetry_as_test,
    telemetry_instance,
)
from haystack.testing.test_utils import set_all_seeds

set_all_seeds(0)


@pytest.fixture
def samples_path():
    return Path(__file__).parent / "samples"


@pytest.fixture
def del_hf_env_vars(monkeypatch):
    """
    Delete Hugging Face environment variables for tests.

    Prevents passing empty tokens to Hugging Face, which would cause API calls to fail.
    This is particularly relevant for PRs opened from forks, where secrets are not available
    and empty environment variables might be set instead of being removed.

    See https://github.com/deepset-ai/haystack/issues/8811 for more details.
    """
    monkeypatch.delenv("HF_API_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
