# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import time
from pathlib import Path
from typing import Any

from haystack import component, default_from_dict, default_to_dict, logging

logger = logging.getLogger(__name__)


@component
class GNAPResultReader:
    """
    Reads results from a GNAP (Git-Native Agent Protocol) task board.

    Polls ``board/done/<task_id>.md`` until the file appears or the timeout expires.

    ### Usage example

    ```python
    from haystack.components.coordinators import GNAPResultReader

    reader = GNAPResultReader(board_path="/path/to/board", timeout=60)
    result = reader.run(task_id="<uuid>")
    print(result["result"])
    ```
    """

    def __init__(self, board_path: str, timeout: float = 300.0, poll_interval: float = 1.0):
        """
        Initialize the GNAPResultReader.

        :param board_path: Path to the git repository used as the task board.
        :param timeout: Maximum seconds to wait for the result file. Defaults to 300.
        :param poll_interval: Seconds between polling attempts. Defaults to 1.0.
        """
        self.board_path = board_path
        self.timeout = timeout
        self.poll_interval = poll_interval

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(self, board_path=self.board_path, timeout=self.timeout, poll_interval=self.poll_interval)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GNAPResultReader":
        return default_from_dict(cls, data)

    @component.output_types(result=str)
    def run(self, task_id: str) -> dict[str, Any]:
        """
        Wait for ``board/done/<task_id>.md`` to appear and return its content.

        :param task_id: The task identifier returned by :class:`GNAPTaskCreator`.
        :returns: A dict with ``result`` (str) containing the task result file content.
        :raises TimeoutError: If the result file does not appear within ``timeout`` seconds.
        """
        done_file = Path(self.board_path) / "board" / "done" / f"{task_id}.md"
        deadline = time.monotonic() + self.timeout

        while time.monotonic() < deadline:
            if done_file.exists():
                content = done_file.read_text(encoding="utf-8")
                logger.debug("GNAP task '{task_id}' completed, read {size} bytes", task_id=task_id, size=len(content))
                return {"result": content}
            time.sleep(self.poll_interval)

        msg = f"GNAP task '{task_id}' did not complete within {self.timeout}s"
        raise TimeoutError(msg)
