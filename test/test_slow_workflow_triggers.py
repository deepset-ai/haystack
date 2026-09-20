# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import re
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOW = _ROOT / ".github" / "workflows" / "slow.yml"


def _trigger_paths() -> list[str]:
    """
    Return the file paths listed in the `changes` filter of the Slow Integration Tests workflow.

    These are the paths dorny/paths-filter compares the PR diff against; a slow test module that is not
    listed here never triggers the workflow.
    """
    workflow = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["check-if-changed"]["steps"]
    raw_filters = next(step["with"]["filters"] for step in steps if step.get("with", {}).get("filters"))
    # `filters` is an embedded YAML string: dorny/paths-filter parses it the same way.
    return yaml.safe_load(raw_filters)["changes"]


def _slow_test_modules() -> list[Path]:
    """Return every test module under `test/` that declares at least one `@pytest.mark.slow` test."""
    marker = re.compile(r"^\s*@pytest\.mark\.slow\b", re.MULTILINE)
    return sorted(
        path.relative_to(_ROOT)
        for path in (_ROOT / "test").rglob("test_*.py")
        if marker.search(path.read_text("utf-8"))
    )


def _haystack_module_of(test_module: Path) -> Path:
    """
    Return the source module a test module covers, following the mirrored `test/` - `haystack/` layout.

    Example: `test/components/evaluators/test_sas_evaluator.py` -> `haystack/components/evaluators/sas_evaluator.py`
    """
    relative_parts = test_module.parts[1:]
    file_name = relative_parts[-1].removeprefix("test_")
    return Path("haystack").joinpath(*relative_parts[:-1], file_name)


class TestSlowWorkflowTriggers:
    def test_every_listed_path_exists(self):
        missing = [path for path in _trigger_paths() if not (_ROOT / path).exists()]
        assert missing == [], (
            ".github/workflows/slow.yml triggers on paths that no longer exist, so those entries never match "
            f"a PR diff: {missing}"
        )

    def test_every_slow_test_module_and_its_source_trigger_the_workflow(self):
        listed = set(_trigger_paths())
        not_triggered: list[str] = []
        for test_module in _slow_test_modules():
            haystack_module = _haystack_module_of(test_module)
            assert haystack_module.exists(), (
                f"{test_module} declares slow tests but {haystack_module} is missing: the mirrored module name "
                "no longer matches the test module, so update this test to resolve it."
            )
            # CONTRIBUTING.md: "If you mark a test but do not include both the test file and the file to be
            # tested in the list, the test won't run automatically."
            for path in (test_module.as_posix(), haystack_module.as_posix()):
                if path not in listed:
                    not_triggered.append(path)
        assert not_triggered == [], (
            "These files hold slow integration tests that no PR can trigger, because they are missing from the "
            f"`changes` filter of .github/workflows/slow.yml: {not_triggered}"
        )
