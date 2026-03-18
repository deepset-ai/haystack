# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from haystack.components.coordinators.gnap_result_reader import GNAPResultReader


class TestGNAPResultReader:
    def test_run_reads_existing_result(self, tmp_path):
        done_dir = tmp_path / "board" / "done"
        done_dir.mkdir(parents=True)
        task_id = "test-task-123"
        (done_dir / f"{task_id}.md").write_text("# Result\nDone!", encoding="utf-8")

        reader = GNAPResultReader(board_path=str(tmp_path))
        result = reader.run(task_id=task_id)

        assert result["result"] == "# Result\nDone!"

    def test_run_times_out_when_no_result(self, tmp_path):
        reader = GNAPResultReader(board_path=str(tmp_path), timeout=0.05, poll_interval=0.01)
        with pytest.raises(TimeoutError, match="test-missing"):
            reader.run(task_id="test-missing")

    def test_run_waits_for_result(self, tmp_path):
        import threading

        done_dir = tmp_path / "board" / "done"
        done_dir.mkdir(parents=True)
        task_id = "async-task"

        def write_result():
            import time

            time.sleep(0.05)
            (done_dir / f"{task_id}.md").write_text("result content", encoding="utf-8")

        thread = threading.Thread(target=write_result)
        thread.start()

        reader = GNAPResultReader(board_path=str(tmp_path), timeout=5.0, poll_interval=0.01)
        result = reader.run(task_id=task_id)
        thread.join()

        assert result["result"] == "result content"

    def test_to_dict(self, tmp_path):
        reader = GNAPResultReader(board_path=str(tmp_path), timeout=120.0, poll_interval=2.0)
        data = reader.to_dict()
        assert data["init_parameters"]["board_path"] == str(tmp_path)
        assert data["init_parameters"]["timeout"] == 120.0
        assert data["init_parameters"]["poll_interval"] == 2.0

    def test_from_dict(self, tmp_path):
        data = {
            "type": "haystack.components.coordinators.gnap_result_reader.GNAPResultReader",
            "init_parameters": {"board_path": str(tmp_path), "timeout": 60.0, "poll_interval": 0.5},
        }
        reader = GNAPResultReader.from_dict(data)
        assert reader.board_path == str(tmp_path)
        assert reader.timeout == 60.0
        assert reader.poll_interval == 0.5
