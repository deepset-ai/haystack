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
        :param root: Directory under which result files are written. Created on first write if it does not exist.
        """
        self.root = Path(root)

    def write(self, *, key: str, content: str) -> str:
        """Write `content` to `<root>/<key>` (creating parent directories) and return the file path as a string."""
        path = self.root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return str(path)

    def read(self, reference: str) -> str:
        """Read back the content of the file at `reference`."""
        return Path(reference).read_text(encoding="utf-8")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the store, storing its root directory as a string."""
        return default_to_dict(self, root=str(self.root))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FileSystemToolResultStore":
        """Deserialize the store from a dictionary."""
        return default_from_dict(cls, data)
