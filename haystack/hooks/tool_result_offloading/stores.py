# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any

from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.hooks.tool_result_offloading.types import ToolResultStore


class FileSystemToolResultStore(ToolResultStore):
    """
    A `ToolResultStore` that writes offloaded tool results to files under a root directory on the local file system.

    ```python
    from haystack.hooks.tool_result_offloading import FileSystemToolResultStore

    store = FileSystemToolResultStore(root="tool_results")
    reference = store.write(key="search_1.txt", content="...")
    store.read(reference)
    ```

    Binary content is supported too: `write` takes bytes (an offloaded image or file) and `read` returns them
    unchanged, while text content round trips as a string.
    """

    def __init__(self, root: str | Path) -> None:
        """
        Initialize the store with the root directory results are written under.

        :param root: Directory under which result files are written. Created on first write if it does not exist.
        """
        self.root = Path(root)

    def _resolve_in_root(self, path_like: str | Path, *, subject: str) -> Path:
        """
        Resolve a path-like value and ensure it stays within the configured store root.

        Relative values are interpreted relative to `self.root`; absolute values are used as-is.

        :param path_like: Relative or absolute path-like value to resolve.
        :param subject: Human-readable label used in the error message.
        :returns: The resolved absolute path within the store root.
        :raises ValueError: If the resolved path escapes the store root.
        """
        root = self.root.resolve()
        path = Path(path_like)
        candidate = path if path.is_absolute() else root / path
        resolved = candidate.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f"{subject} '{path_like}' resolves outside the store root '{root}'.")
        return resolved

    def write(self, *, key: str, content: str | bytes) -> str:
        """
        Write `content` to `<root>/<key>`, creating parent directories, and return the file path.

        Text is written UTF-8 encoded; bytes (an offloaded image or file) are written verbatim.

        The resolved target must stay within the root directory: a `key` that escapes it (e.g. containing `../` or an
        absolute path) is rejected, so a tool-provided key cannot write outside the store.

        :param key: Relative file name for the result within the store root.
        :param content: The tool result to persist, as text or as raw bytes.
        :returns: The absolute path the content was written to, as a string, for use with `read`.
        :raises ValueError: If `key` resolves to a location outside the store root.
        """
        path = self._resolve_in_root(key, subject="Result key")
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")
        return str(path)

    def read(self, reference: str) -> str | bytes:
        """
        Read back the content previously written to `reference`.

        A file whose bytes are valid UTF-8 is returned as a string, so text results round trip unchanged; anything
        else (an offloaded image or file) is returned as raw bytes.

        The resolved reference must stay within the store root: it is a store-scoped reference returned by `write`,
        to be passed back unchanged, not an arbitrary filesystem path callers can build themselves.

        :param reference: A store reference returned by `write`.
        :returns: The stored content, as text when it decodes as UTF-8 and as bytes otherwise.
        :raises ValueError: If `reference` resolves to a location outside the store root.
        """
        data = self._resolve_in_root(reference, subject="Result reference").read_bytes()
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return data

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the store, storing its root directory as a string.

        :returns: A dictionary representation of the store.
        """
        return default_to_dict(self, root=str(self.root))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FileSystemToolResultStore":
        """
        Deserialize the store from a dictionary.

        :param data: A dictionary representation produced by `to_dict`.
        :returns: The deserialized `FileSystemToolResultStore`.
        """
        return default_from_dict(cls, data)
