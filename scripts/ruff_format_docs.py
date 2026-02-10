"""
Pre-commit hook that runs ruff format on Python code blocks in Markdown/MDX files.

Uses the ruff configuration from pyproject.toml automatically.
"""

import os
import re
import subprocess
import sys
import tempfile

PYTHON_FENCE_RE = re.compile(
    r"(?P<before>^```python\s*\n)"
    r"(?P<code>.*?)"
    r"(?P<after>^```\s*$)",
    re.MULTILINE | re.DOTALL,
)


def _find_tool(name: str) -> str:
    """Find a tool installed in the same virtualenv as the running Python."""
    return os.path.join(os.path.dirname(sys.executable), name)


def _ruff(code: str) -> str:
    return subprocess.run(
        [
            _find_tool("ruff"),
            "format",
            "--config",
            "format.skip-magic-trailing-comma = false",
            "--stdin-filename",
            "block.py",
            "-",
        ],
        input=code,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _add_trailing_commas(code: str) -> str:
    """Add trailing commas to multi-line expressions using add-trailing-comma."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmpfile = f.name
    try:
        subprocess.run([_find_tool("add-trailing-comma"), tmpfile], capture_output=True, check=False)
        with open(tmpfile) as f:
            return f.read()
    finally:
        os.unlink(tmpfile)


def format_code_block(match: re.Match) -> str:
    """Format a single code block"""
    code = match.group("code")
    try:
        # 1. ruff format (may create new multi-line expressions)
        # 2. add trailing commas to all multi-line expressions
        # 3. ruff format again (respects trailing commas, ensures stable output)
        formatted = _ruff(_add_trailing_commas(_ruff(code)))
    except subprocess.CalledProcessError:
        # If ruff can't parse the snippet (e.g. partial code), leave it unchanged.
        return match.group(0)
    return match.group("before") + formatted + match.group("after")


def main() -> int:
    """Main entrypoint"""
    ret = 0
    for path in sys.argv[1:]:
        with open(path) as f:
            original = f.read()
        new = PYTHON_FENCE_RE.sub(format_code_block, original)
        if new != original:
            with open(path, "w") as f:
                f.write(new)
            print(f"{path}: Rewriting...")
            ret = 1
    return ret


if __name__ == "__main__":
    raise SystemExit(main())
