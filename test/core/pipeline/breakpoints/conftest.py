# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from haystack.core.pipeline import Pipeline
from haystack.core.pipeline.breakpoint import HAYSTACK_PIPELINE_SNAPSHOT_SAVE_ENABLED, load_pipeline_snapshot


@pytest.fixture(autouse=True)
def enable_snapshot_saving(monkeypatch):
    """Enable snapshot file saving for these integration tests."""
    monkeypatch.setenv(HAYSTACK_PIPELINE_SNAPSHOT_SAVE_ENABLED, "true")


@pytest.fixture
def output_directory(tmp_path):
    """Provide a temporary directory for snapshot files."""
    return tmp_path


@pytest.fixture
def load_and_resume_pipeline_snapshot():
    """Fixture that returns a function to load and resume a pipeline from a snapshot."""

    def _resume(pipeline: Pipeline, output_directory: Path, component_name: str, data: dict | None = None) -> dict:
        """
        Utility function to load and resume pipeline snapshot from a breakpoint file.

        :param pipeline: The pipeline instance to resume
        :param output_directory: Directory containing the breakpoint files
        :param component_name: Component name to look for in breakpoint files
        :param data: Data to pass to the pipeline run (defaults to empty dict)

        :returns:
            Dict containing the pipeline run results

        :raises:
            ValueError: If no breakpoint file is found for the given component
        """
        data = data or {}
        all_files = list(output_directory.glob("*"))

        for full_path in all_files:
            f_name = Path(full_path).name
            if str(f_name).startswith(component_name):
                pipeline_snapshot = load_pipeline_snapshot(full_path)
                return pipeline.run(data=data, pipeline_snapshot=pipeline_snapshot)

        msg = f"No files found for {component_name} in {output_directory}."
        raise ValueError(msg)

    return _resume
