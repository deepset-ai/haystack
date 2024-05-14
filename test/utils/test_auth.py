# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os

import pytest

from haystack.utils.auth import Secret, EnvVarSecret, SecretType, TokenSecret
from dataclasses import FrozenInstanceError


def test_secret_type():
    for e in SecretType:
        assert e == SecretType.from_str(e.value)

    with pytest.raises(ValueError, match="Unknown secret type"):
        SecretType.from_str("disk")


def test_token_secret():
    secret = Secret.from_token("test-token")
    assert secret.type == SecretType.TOKEN
    assert isinstance(secret, TokenSecret)
    assert secret._token == "test-token"
    assert secret.resolve_value() == "test-token"

    with pytest.raises(ValueError, match="Cannot serialize token-based secret"):
        secret.to_dict()

    with pytest.raises(ValueError, match="cannot be empty"):
        Secret.from_token("")

    with pytest.raises(FrozenInstanceError):
        secret._token = "abba"
    with pytest.raises(FrozenInstanceError):
        secret._type = SecretType.ENV_VAR


def test_env_var_secret():
    secret = Secret.from_env_var("TEST_ENV_VAR1")
    os.environ["TEST_ENV_VAR1"] = "test-token"

    assert secret.type == SecretType.ENV_VAR
    assert isinstance(secret, EnvVarSecret)
    assert secret._env_vars == ("TEST_ENV_VAR1",)
    assert secret._strict is True
    assert secret.resolve_value() == "test-token"

    del os.environ["TEST_ENV_VAR1"]
    with pytest.raises(ValueError, match="None of the following .* variables are set"):
        secret.resolve_value()

    secret = Secret.from_env_var("TEST_ENV_VAR2", strict=False)
    assert secret._strict is False
    assert secret.resolve_value() == None

    secret = Secret.from_env_var(["TEST_ENV_VAR2", "TEST_ENV_VAR1"], strict=True)
    assert secret._env_vars == ("TEST_ENV_VAR2", "TEST_ENV_VAR1")
    with pytest.raises(ValueError, match="None of the following .* variables are set"):
        secret.resolve_value()
    os.environ["TEST_ENV_VAR1"] = "test-token-2"
    assert secret.resolve_value() == "test-token-2"
    os.environ["TEST_ENV_VAR2"] = "test-token"
    assert secret.resolve_value() == "test-token"

    with pytest.raises(ValueError, match="One or more environment variables"):
        Secret.from_env_var([])

    assert secret.to_dict() == {"type": "env_var", "env_vars": ["TEST_ENV_VAR2", "TEST_ENV_VAR1"], "strict": True}
    assert (
        Secret.from_dict({"type": "env_var", "env_vars": ["TEST_ENV_VAR2", "TEST_ENV_VAR1"], "strict": True}) == secret
    )

    with pytest.raises(FrozenInstanceError):
        secret._env_vars = ("A", "B")
    with pytest.raises(FrozenInstanceError):
        secret._strict = False
    with pytest.raises(FrozenInstanceError):
        secret._type = SecretType.TOKEN
