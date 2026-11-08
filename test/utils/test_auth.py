# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import FrozenInstanceError

import pytest

from haystack.utils.auth import EnvVarSecret, Secret, SecretType, TokenSecret, deserialize_secrets_inplace


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
        secret._token = "abba"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        secret._type = SecretType.ENV_VAR  # type: ignore[misc]

    secret = Secret.from_token("sk-supersecret-1234567890ABCDEF")
    assert "sk-supersecret-1234567890ABCDEF" not in repr(secret)
    assert "sk-supersecret-1234567890ABCDEF" not in str(secret)
    assert "<redacted>" in repr(secret)


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
    assert isinstance(secret, EnvVarSecret)
    assert secret._strict is False
    assert secret.resolve_value() is None

    secret = Secret.from_env_var(["TEST_ENV_VAR2", "TEST_ENV_VAR1"], strict=True)
    assert isinstance(secret, EnvVarSecret)
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
        secret._env_vars = ("A", "B")  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        secret._strict = False  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        secret._type = SecretType.TOKEN  # type: ignore[misc]


def test_deserialize_secrets_inplace_deserializes_listed_keys():
    data = {"api_key": Secret.from_env_var("TEST_ENV_VAR1").to_dict(), "model": "gpt"}

    deserialize_secrets_inplace(data, ["api_key"])

    assert isinstance(data["api_key"], EnvVarSecret)
    assert data["model"] == "gpt"


def test_deserialize_secrets_inplace_deserializes_listed_keys_when_recursive():
    """A serialized secret is itself a dict, so recursion must not take priority over the requested keys."""
    data = {"api_key": Secret.from_env_var("TEST_ENV_VAR1").to_dict()}

    deserialize_secrets_inplace(data, ["api_key"], recursive=True)

    assert isinstance(data["api_key"], EnvVarSecret)


def test_deserialize_secrets_inplace_recurses_into_nested_dicts():
    data = {"a": {"b": {"api_key": Secret.from_env_var("TEST_ENV_VAR1").to_dict()}}}

    deserialize_secrets_inplace(data, ["api_key"], recursive=True)

    assert isinstance(data["a"]["b"]["api_key"], EnvVarSecret)


def test_deserialize_secrets_inplace_leaves_non_secret_nested_dicts_untouched():
    data = {"a": {"b": {"timeout": 3}}, "api_key": None}

    deserialize_secrets_inplace(data, ["api_key"], recursive=True)

    assert data == {"a": {"b": {"timeout": 3}}, "api_key": None}
