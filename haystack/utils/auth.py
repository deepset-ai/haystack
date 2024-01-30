from enum import Enum
import os
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod


class AuthPolicyType(Enum):
    TOKEN = "token"
    ENV_VAR = "env_var"

    def __str__(self):
        return self.value

    @staticmethod
    def from_str(string: str) -> "AuthPolicyType":
        map = {e.value: e for e in AuthPolicyType}
        type = map.get(string)
        if type is None:
            raise ValueError(f"Unknown authentication policy type '{string}'")
        return type


@dataclass
class AuthPolicy(ABC):
    """
    Provides a common interface for authentication policies.
    """

    _type: AuthPolicyType

    def __init__(self, type: AuthPolicyType):
        super().__init__()
        self._type = type

    @staticmethod
    def from_token(token: str) -> "AuthPolicy":
        """
        Create a token authentication policy.
        This policy cannot be serialized.

        :param token:
            The token to use for authentication.
        """
        return AuthPolicyToken(token)

    @staticmethod
    def from_env_var(env_vars: Union[str, List[str]], *, strict: bool = True) -> "AuthPolicy":
        """
        Create an environment variable authentication policy.
        Accepts one or more environment variables and fetches
        a string token from the first one that is set.

        :param env_vars:
            A single environment variable or an ordered list of
            candidate environment variables.
        :param strict:
            Whether to raise an exception if none of the environment
            variables are set.
        """
        if isinstance(env_vars, str):
            env_vars = [env_vars]
        return AuthPolicyEnvVar(env_vars, strict=strict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the policy to a JSON-serializable dictionary.

        :returns:
            The serialized policy.
        """
        out = {"type": self._type.value}
        inner = self._to_dict()
        assert all(k not in inner for k in out.keys())
        out.update(inner)
        return out

    @staticmethod
    def from_dict(dict: Dict[str, Any]) -> "AuthPolicy":
        """
        Create a policy from a JSON-serializable dictionary.

        :param dict:
            The dictionary to create the policy from.
        :returns:
            The deserialized policy.
        """
        policy_map = {AuthPolicyType.TOKEN: AuthPolicyToken, AuthPolicyType.ENV_VAR: AuthPolicyEnvVar}
        policy_type = AuthPolicyType.from_str(dict["type"])
        return policy_map[policy_type]._from_dict(dict)  # type: ignore

    @abstractmethod
    def resolve_value(self) -> Optional[Any]:
        """
        Resolve the policy to an atomic value. The semantics
        of the value is policy-dependent.

        :returns:
            The value of the policy, if any.
        """
        pass

    @abstractmethod
    def _to_dict(self) -> Dict[str, Any]:
        pass

    @staticmethod
    @abstractmethod
    def _from_dict(dict: Dict[str, Any]) -> "AuthPolicy":
        pass


@dataclass
class AuthPolicyToken(AuthPolicy):
    """
    An authentication policy that uses a string token/API key.
    This policy cannot be serialized.
    """

    _token: str

    def __init__(self, token: str):
        """
        Create a token authentication policy.

        :param token:
            The token to use for authentication.
        """
        super().__init__(AuthPolicyType.TOKEN)
        self._token = token

        if len(token) == 0:
            raise ValueError("Authentication token cannot be empty.")

    def _to_dict(self) -> Dict[str, Any]:
        raise ValueError(
            "Cannot serialize token authentication policy. Use an alternative policy like environment variables."
        )

    @staticmethod
    def _from_dict(dict: Dict[str, Any]) -> "AuthPolicy":
        raise ValueError(
            "Cannot deserialize token authentication policy. Use an alternative policy like environment variables."
        )

    def resolve_value(self) -> Optional[Any]:
        return self._token


@dataclass
class AuthPolicyEnvVar(AuthPolicy):
    """
    An authentication policy that accepts one or more
    environment variables and fetches a string token from
    the first one that is set.
    """

    _env_vars: List[str]
    _strict: bool

    def __init__(self, env_vars: List[str], *, strict: bool = True):
        """
        Create an environment variable authentication policy.

        :param env_vars:
            Ordered list of candidate environment variables.
        :param strict:
            Whether to raise an exception if none of the environment
            variables are set.
        """
        super().__init__(AuthPolicyType.ENV_VAR)
        self._env_vars = list(env_vars)
        self._strict = strict

        if len(env_vars) == 0:
            raise ValueError("One or more environment variables must be provided for the authentication policy.")

    def _to_dict(self) -> Dict[str, Any]:
        return {"env_vars": self._env_vars, "strict": self._strict}

    @staticmethod
    def _from_dict(dict: Dict[str, Any]) -> "AuthPolicy":
        return AuthPolicyEnvVar(dict["env_vars"], strict=dict["strict"])

    def resolve_value(self) -> Optional[Any]:
        out = None
        for env_var in self._env_vars:
            value = os.getenv(env_var)
            if value is not None:
                out = value
                break
        if out is None and self._strict:
            raise ValueError(f"None of the following authentication environment variables are set: {self._env_vars}")
        return out
