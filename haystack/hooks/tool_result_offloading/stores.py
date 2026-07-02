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
    """

    def __init__(self, root: str | Path) -> None:
        """
        Initialize the store with the root directory results are written under.

        :param root: Directory under which result files are written. Created on first write if it does not exist.
        """
        self.root = Path(root)

    def write(self, *, key: str, content: str) -> str:
        """
        Write `content` to `<root>/<key>`, creating parent directories, and return the file path.

        The resolved target must stay within the root directory: a `key` that escapes it (e.g. containing `../` or an
        absolute path) is rejected, so a tool-provided key cannot write outside the store.

        :param key: Relative file name for the result within the store root.
        :param content: The tool result to persist.
        :returns: The absolute path the content was written to, as a string, for use with `read`.
        :raises ValueError: If `key` resolves to a location outside the store root.
        """
        root = self.root.resolve()
        path = (root / key).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Result key '{key}' resolves outside the store root '{root}'.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return str(path)

    def read(self, reference: str) -> str:
        """
        Read back the content previously written to `reference`.

        :param reference: A path returned by `write`.
        :returns: The stored content.
        """
        return Path(reference).read_text(encoding="utf-8")

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
