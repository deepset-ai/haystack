# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import subprocess
import uuid
from pathlib import Path
from typing import Any, Optional

from haystack import component, default_from_dict, default_to_dict, logging

logger = logging.getLogger(__name__)


@component
class GNAPTaskCreator:
    """
    Creates tasks in a GNAP (Git-Native Agent Protocol) task board.

    GNAP uses a git repository as a task board where tasks move through
    ``board/todo/`` → ``board/doing/`` → ``board/done/`` as markdown files.
    Each transition is recorded as a git commit.

    ### Usage example

    ```python
    from haystack.components.coordinators import GNAPTaskCreator

    creator = GNAPTaskCreator(board_path="/path/to/board")
    result = creator.run(query="Summarize this document", task_spec={"priority": "high"})
    print(result["task_id"])
    ```
    """

    def __init__(self, board_path: str, use_git: bool = True):
        """
        Initialize the GNAPTaskCreator.

        :param board_path: Path to the git repository used as the task board.
        :param use_git: Whether to commit task creation to git. Defaults to True.
        """
        self.board_path = board_path
        self.use_git = use_git

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(self, board_path=self.board_path, use_git=self.use_git)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GNAPTaskCreator":
        return default_from_dict(cls, data)

    @component.output_types(task_id=str)
    def run(self, query: str, task_spec: Optional[dict] = None) -> dict[str, Any]:
        """
        Write a task markdown file to ``board/todo/`` and optionally git-commit it.

        :param query: The task description or query.
        :param task_spec: Optional dictionary of additional task metadata.
        :returns: A dict with ``task_id`` (str) identifying the created task.
        """
        task_id = str(uuid.uuid4())
        todo_dir = Path(self.board_path) / "board" / "todo"
        todo_dir.mkdir(parents=True, exist_ok=True)

        task_file = todo_dir / f"{task_id}.md"
        content_lines = [f"# Task {task_id}", "", f"**Query:** {query}", ""]
        if task_spec:
            content_lines.append("## Spec")
            for key, value in task_spec.items():
                content_lines.append(f"- **{key}:** {value}")
            content_lines.append("")
        task_file.write_text("\n".join(content_lines), encoding="utf-8")

        if self.use_git:
            self._git_commit(task_file, f"gnap: create task {task_id}")

        logger.debug("Created GNAP task '{task_id}' at {task_file}", task_id=task_id, task_file=task_file)
        return {"task_id": task_id}

    def _git_commit(self, file_path: Path, message: str) -> None:
        board = Path(self.board_path)
        subprocess.run(["git", "add", str(file_path)], cwd=board, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", message], cwd=board, check=True, capture_output=True)
