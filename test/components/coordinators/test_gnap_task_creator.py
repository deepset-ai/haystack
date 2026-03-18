# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import patch

import pytest

from haystack.components.coordinators.gnap_task_creator import GNAPTaskCreator


class TestGNAPTaskCreator:
    def test_run_creates_task_file(self, tmp_path):
        creator = GNAPTaskCreator(board_path=str(tmp_path), use_git=False)
        result = creator.run(query="Summarize the document")

        task_id = result["task_id"]
        assert isinstance(task_id, str)
        task_file = tmp_path / "board" / "todo" / f"{task_id}.md"
        assert task_file.exists()

    def test_run_task_file_contains_query(self, tmp_path):
        creator = GNAPTaskCreator(board_path=str(tmp_path), use_git=False)
        result = creator.run(query="My test query")

        task_file = tmp_path / "board" / "todo" / f"{result['task_id']}.md"
        content = task_file.read_text(encoding="utf-8")
        assert "My test query" in content

    def test_run_task_file_contains_spec(self, tmp_path):
        creator = GNAPTaskCreator(board_path=str(tmp_path), use_git=False)
        result = creator.run(query="query", task_spec={"priority": "high", "model": "gpt-4"})

        task_file = tmp_path / "board" / "todo" / f"{result['task_id']}.md"
        content = task_file.read_text(encoding="utf-8")
        assert "priority" in content
        assert "high" in content

    def test_run_without_task_spec(self, tmp_path):
        creator = GNAPTaskCreator(board_path=str(tmp_path), use_git=False)
        result = creator.run(query="query")
        assert "task_id" in result

    def test_run_with_git(self, tmp_path):
        creator = GNAPTaskCreator(board_path=str(tmp_path), use_git=True)
        with patch("subprocess.run") as mock_run:
            result = creator.run(query="query")
        assert mock_run.call_count == 2  # git add + git commit
        assert "task_id" in result

    def test_to_dict(self, tmp_path):
        creator = GNAPTaskCreator(board_path=str(tmp_path), use_git=False)
        data = creator.to_dict()
        assert data["init_parameters"]["board_path"] == str(tmp_path)
        assert data["init_parameters"]["use_git"] is False

    def test_from_dict(self, tmp_path):
        data = {
            "type": "haystack.components.coordinators.gnap_task_creator.GNAPTaskCreator",
            "init_parameters": {"board_path": str(tmp_path), "use_git": False},
        }
        creator = GNAPTaskCreator.from_dict(data)
        assert creator.board_path == str(tmp_path)
        assert creator.use_git is False

    def test_multiple_tasks_get_unique_ids(self, tmp_path):
        creator = GNAPTaskCreator(board_path=str(tmp_path), use_git=False)
        ids = {creator.run(query="q")["task_id"] for _ in range(5)}
        assert len(ids) == 5
