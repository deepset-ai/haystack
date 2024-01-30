import os

import pytest

from haystack.utils.auth import AuthPolicy, AuthPolicyEnvVar, AuthPolicyType, AuthPolicyToken


def test_auth_policy_type():
    for e in AuthPolicyType:
        assert e == AuthPolicyType.from_str(e.value)

    with pytest.raises(ValueError, match="Unknown authentication policy type"):
        AuthPolicyType.from_str("disk")


def test_auth_policy_token():
    policy = AuthPolicy.from_token("test-token")
    assert policy._type == AuthPolicyType.TOKEN
    assert isinstance(policy, AuthPolicyToken)
    assert policy._token == "test-token"
    assert policy.resolve_value() == "test-token"

    with pytest.raises(ValueError, match="Cannot serialize token authentication policy"):
        policy.to_dict()

    with pytest.raises(ValueError, match="cannot be empty"):
        AuthPolicy.from_token("")


def test_auth_policy_env_var():
    policy = AuthPolicy.from_env_var("TEST_ENV_VAR1")
    os.environ["TEST_ENV_VAR1"] = "test-token"

    assert policy._type == AuthPolicyType.ENV_VAR
    assert isinstance(policy, AuthPolicyEnvVar)
    assert policy._env_vars == ["TEST_ENV_VAR1"]
    assert policy._strict is True
    assert policy.resolve_value() == "test-token"

    del os.environ["TEST_ENV_VAR1"]
    with pytest.raises(ValueError, match="None of the following .* variables are set"):
        policy.resolve_value()

    policy = AuthPolicy.from_env_var("TEST_ENV_VAR2", strict=False)
    assert policy._strict is False
    assert policy.resolve_value() == None

    policy = AuthPolicy.from_env_var(["TEST_ENV_VAR2", "TEST_ENV_VAR1"], strict=True)
    assert policy._env_vars == ["TEST_ENV_VAR2", "TEST_ENV_VAR1"]
    with pytest.raises(ValueError, match="None of the following .* variables are set"):
        policy.resolve_value()
    os.environ["TEST_ENV_VAR1"] = "test-token-2"
    assert policy.resolve_value() == "test-token-2"
    os.environ["TEST_ENV_VAR2"] = "test-token"
    assert policy.resolve_value() == "test-token"

    with pytest.raises(ValueError, match="One or more environment variables"):
        AuthPolicy.from_env_var([])

    assert policy.to_dict() == {"type": "env_var", "env_vars": ["TEST_ENV_VAR2", "TEST_ENV_VAR1"], "strict": True}
    assert (
        AuthPolicy.from_dict({"type": "env_var", "env_vars": ["TEST_ENV_VAR2", "TEST_ENV_VAR1"], "strict": True})
        == policy
    )
