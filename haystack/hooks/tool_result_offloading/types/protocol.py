# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Protocol

from haystack.components.agents.state.state import State
from haystack.core.serialization import default_from_dict, default_to_dict


class ToolResultStore(Protocol):
    """
    A place a `ToolResultOffloadHook` writes offloaded tool results to, and reads them back from.

    Implementations decide where and how the content lives (local disk, an isolated sandbox filesystem, object
    storage, ...). `write` returns an opaque reference string that the Agent puts in the conversation in place of the
    full result; `read` resolves that reference back to the original content.

    Implement both `to_dict` and `from_dict` to make a custom store serializable; the default implementations below
    cover stores whose constructor takes no arguments.
    """

    def write(self, *, key: str, content: str) -> str:
        """
        Persist `content` under `key` and return an opaque reference to it.

        :param key: A stable, per-result identifier the hook derives from the tool call (e.g. a file name).
        :param content: The tool result to persist.
        :returns: A reference string (e.g. a path or URI) that `read` can later resolve.
        """
        ...

    def read(self, reference: str) -> str:
        """Return the content previously stored under `reference`."""
        ...

    def to_dict(self) -> dict[str, Any]:
        """Serialize the store to a dictionary."""
        return default_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolResultStore":
        """Deserialize the store from a dictionary."""
        return default_from_dict(cls, data)


class OffloadPolicy(Protocol):
    """
    Decides, per tool result, whether the `ToolResultOffloadHook` offloads it to the store or leaves it in context.

    A `ToolResultOffloadHook` maps tool names to policies, so different tools can offload under different conditions
    (always, never, or a custom rule such as a size threshold).

    Implement both `to_dict` and `from_dict` to make a custom policy serializable; the default implementations below
    cover policies whose constructor takes no arguments.
    """

    def should_offload(self, tool_name: str, result: str, state: State) -> bool:
        """
        Return whether the given tool result should be offloaded.

        :param tool_name: The name of the tool that produced the result.
        :param result: The tool result as a string (the content that would otherwise stay in the conversation).
        :param state: The Agent's live `State`, for policies that decide based on run context.
        :returns: True to offload the result to the store, False to leave it in context.
        """
        ...

    def to_dict(self) -> dict[str, Any]:
        """Serialize the policy to a dictionary."""
        return default_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OffloadPolicy":
        """Deserialize the policy from a dictionary."""
        return default_from_dict(cls, data)
