# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import re
from pathlib import Path


def test_requires_python_marks_py314_as_unsupported():
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    pyproject_text = pyproject_path.read_text(encoding="utf-8")

    match = re.search(r'^requires-python\s*=\s*"([^"]+)"', pyproject_text, flags=re.MULTILINE)

    assert match is not None
    assert match.group(1) == ">=3.10,<3.14"
