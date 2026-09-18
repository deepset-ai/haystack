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
    storage, ...). `write` returns a reference string that the Agent puts in the conversation in place of the full
    result; `read` resolves that reference back to the original content. Only the store interprets a reference -
    callers pass it back to `read` unchanged.

    A store that sets `supports_binary_content` takes bytes in `write` and gives them back from `read`. One that
    leaves it False is only ever given text, and image and file results stay in the conversation instead.

    Implement both `to_dict` and `from_dict` to make a custom store serializable; the default implementations below
    cover stores whose constructor takes no arguments.
    """

    # Whether the store handles binary content: `write` accepting bytes and `read` returning them. When False, the
    # store is only ever given text, and the image and file content of a tool result stays in the conversation.
    supports_binary_content: bool = False

    def write(self, *, key: str, content: str | bytes) -> str:
        """
        Persist `content` under `key` and return a reference to it.

        :param key: A stable, per-result identifier the hook derives from the tool call (e.g. a file name). It carries
            an extension matching the content, so a store that maps keys to files can use it as-is.
        :param content: The tool result to persist. Text arrives as a string. Image and file content arrives as the
            decoded bytes of its base64 payload, and only when the store sets `supports_binary_content` to True - a
            text-only store may narrow this parameter to `str`.
        :returns: A reference string (e.g. a path or URI) that `read` can later resolve.
        """
        ...

    def read(self, reference: str) -> str | bytes:
        """
        Return the content previously stored under `reference`.

        :param reference: A reference string returned by `write`.
        :returns: The stored content: a string for content written as text, bytes for binary content such as an
            offloaded image or file. A store that does not support binary content only ever returns a string.
        """
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
        :param result: The tool result as a string (the content that would otherwise stay in the conversation). For a
            result carrying image or file blocks, this is the text and base64 payloads of all its blocks joined
            together, so its length reflects the context the result actually occupies.
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
