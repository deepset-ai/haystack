# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union


class SecretType(Enum):
    TOKEN = "token"
    ENV_VAR = "env_var"

    def __str__(self):
        return self.value

    @staticmethod
    def from_str(string: str) -> "SecretType":
        """
        Convert a string to a SecretType.

        :param string: The string to convert.
        """
        mapping = {e.value: e for e in SecretType}
        _type = mapping.get(string)
        if _type is None:
            raise ValueError(f"Unknown secret type '{string}'")
        return _type


class Secret(ABC):
    """
    Encapsulates a secret used for authentication.

    Usage example:
    ```python
    from haystack.components.generators import OpenAIGenerator
    from haystack.utils import Secret

    generator = OpenAIGenerator(api_key=Secret.from_token("<here_goes_your_token>"))
    ```
    """

    @staticmethod
    def from_token(token: str) -> "Secret":
        """
        Create a token-based secret. Cannot be serialized.

        :param token:
            The token to use for authentication.
        """
        return TokenSecret(_token=token)

    @staticmethod
    def from_env_var(env_vars: Union[str, List[str]], *, strict: bool = True) -> "Secret":
        """
        Create an environment variable-based secret. Accepts one or more environment variables.

        Upon resolution, it returns a string token from the first environment variable that is set.

        :param env_vars:
            A single environment variable or an ordered list of
            candidate environment variables.
        :param strict:
            Whether to raise an exception if none of the environment
            variables are set.
        """
        if isinstance(env_vars, str):
            env_vars = [env_vars]
        return EnvVarSecret(_env_vars=tuple(env_vars), _strict=strict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the secret to a JSON-serializable dictionary.

        Some secrets may not be serializable.

        :returns:
            The serialized policy.
        """
        out = {"type": self.type.value}
        inner = self._to_dict()
        assert all(k not in inner for k in out.keys())
        out.update(inner)
        return out

    @staticmethod
    def from_dict(dict: Dict[str, Any]) -> "Secret":  # noqa:A002
        """
        Create a secret from a JSON-serializable dictionary.

        :param dict:
            The dictionary with the serialized data.
        :returns:
            The deserialized secret.
        """
        secret_map = {SecretType.TOKEN: TokenSecret, SecretType.ENV_VAR: EnvVarSecret}
        secret_type = SecretType.from_str(dict["type"])
        return secret_map[secret_type]._from_dict(dict)  # type: ignore

    @abstractmethod
    def resolve_value(self) -> Optional[Any]:
        """
        Resolve the secret to an atomic value. The semantics of the value is secret-dependent.

        :returns:
            The value of the secret, if any.
        """
        pass

    @property
    @abstractmethod
    def type(self) -> SecretType:
        """
        The type of the secret.
        """
        pass

    @abstractmethod
    def _to_dict(self) -> Dict[str, Any]:
        pass

    @staticmethod
    @abstractmethod
    def _from_dict(_: Dict[str, Any]) -> "Secret":
        pass


@dataclass(frozen=True)
class TokenSecret(Secret):
    """
    A secret that uses a string token/API key.

    Cannot be serialized.
    """

    _token: str
    _type: SecretType = SecretType.TOKEN

    def __post_init__(self):
        super().__init__()
        assert self._type == SecretType.TOKEN

        if len(self._token) == 0:
            raise ValueError("Authentication token cannot be empty.")

    def _to_dict(self) -> Dict[str, Any]:
        raise ValueError(
            "Cannot serialize token-based secret. Use an alternative secret type like environment variables."
        )

    @staticmethod
    def _from_dict(_: Dict[str, Any]) -> "Secret":
        raise ValueError(
            "Cannot deserialize token-based secret. Use an alternative secret type like environment variables."
        )

    def resolve_value(self) -> Optional[Any]:
        """Return the token."""
        return self._token

    @property
    def type(self) -> SecretType:
        """The type of the secret."""
        return self._type


@dataclass(frozen=True)
class EnvVarSecret(Secret):
    """
    A secret that accepts one or more environment variables.

    Upon resolution, it returns a string token from the first environment variable that is set. Can be serialized.
    """

    _env_vars: Tuple[str, ...]
    _strict: bool = True
    _type: SecretType = SecretType.ENV_VAR

    def __post_init__(self):
        super().__init__()
        assert self._type == SecretType.ENV_VAR

        if len(self._env_vars) == 0:
            raise ValueError("One or more environment variables must be provided for the secret.")

    def _to_dict(self) -> Dict[str, Any]:
        return {"env_vars": list(self._env_vars), "strict": self._strict}

    @staticmethod
    def _from_dict(dictionary: Dict[str, Any]) -> "Secret":
        return EnvVarSecret(tuple(dictionary["env_vars"]), _strict=dictionary["strict"])

    def resolve_value(self) -> Optional[Any]:
        """Resolve the secret to an atomic value. The semantics of the value is secret-dependent."""
        out = None
        for env_var in self._env_vars:
            value = os.getenv(env_var)
            if value is not None:
                out = value
                break
        if out is None and self._strict:
            raise ValueError(f"None of the following authentication environment variables are set: {self._env_vars}")
        return out

    @property
    def type(self) -> SecretType:
        """The type of the secret."""
        return self._type


def deserialize_secrets_inplace(data: Dict[str, Any], keys: Iterable[str], *, recursive: bool = False):
    """
    Deserialize secrets in a dictionary inplace.

    :param data:
        The dictionary with the serialized data.
    :param keys:
        The keys of the secrets to deserialize.
    :param recursive:
        Whether to recursively deserialize nested dictionaries.
    """
    for k, v in data.items():
        if isinstance(v, dict) and recursive:
            deserialize_secrets_inplace(v, keys)
        elif k in keys and v is not None:
            data[k] = Secret.from_dict(v)
