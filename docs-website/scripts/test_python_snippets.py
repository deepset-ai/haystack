#!/usr/bin/env python3
"""
Background tester for Python code snippets embedded in Docusaurus Markdown/MDX files.

Features:
- Recursively scans specified directories for .md and .mdx files
- Extracts triple-backtick fenced blocks labeled with "python" or "py"
- Skips blocks preceded by an immediate "<!-- test-ignore -->" marker
- Supports markers above a block:
  - "<!-- test-run -->" to force running even if heuristically considered a concept
  - "<!-- test-concept -->" to force skipping as illustrative
  - "<!-- test-require-files: path1 path2 -->" to require files to exist (skip if missing)
- Optionally skips blocks containing unsafe patterns
- Executes each snippet in isolation via a temporary file using a Python subprocess
- Times out long-running snippets
- Emits GitHub Actions annotations for failures with file and line details
- Summarizes results and sets a non-zero exit code on failures

Usage:
  # Scan default trees
  python scripts/test_python_snippets.py --paths docs versioned_docs --timeout-seconds 30

  # Run a single file (positional target)
  python scripts/test_python_snippets.py docs/concepts/pipelines.mdx

  # Run multiple specific files (positional targets)
  python scripts/test_python_snippets.py docs/overview/intro.mdx docs/concepts/components.mdx

  # Force-run a snippet without imports via marker above the block
  <!-- test-run -->
  ```python
  print("hello world")
  ```

  # Mark an illustrative snippet to skip
  <!-- test-concept -->
  ```python
  @dataclass
  class Foo:
      ...
  ```

  # Require fixtures; snippet will be skipped if files are missing
  <!-- test-require-files: assets/dog.jpg data/example.json -->
  ```python
  from haystack.dataclasses import ByteStream
  image = ByteStream.from_file_path("assets/dog.jpg")
  ```
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional

FENCE_START_RE = re.compile(r"^\s*```(?P<lang>[^\n\r]*)\s*$")
FENCE_END_RE = re.compile(r"^\s*```\s*$")
TEST_IGNORE_MARK = "<!-- test-ignore -->"
TEST_CONCEPT_MARK = "<!-- test-concept -->"
TEST_RUN_MARK = "<!-- test-run -->"
TEST_REQUIRE_FILES_PREFIX = "<!-- test-require-files:"


UNSAFE_PATTERNS = [
    # Basic patterns to avoid obviously unsafe operations in CI examples
    re.compile(r"\bos\.system\s*\("),
    re.compile(r"\bsubprocess\."),
    re.compile(r"\bshutil\.rmtree\s*\("),
    re.compile(r"\bPopen\s*\("),
    re.compile(r"rm\s+-rf\b"),
]


SUPPORTED_EXTENSIONS = {".md", ".mdx"}


@dataclass
class Snippet:
    file_path: str
    """Absolute file path of the Markdown/MDX file."""

    relative_path: str
    """Path relative to the repository root, for nicer output and GH annotations."""

    snippet_index: int
    """Monotonic index of snippet within the file (1-based)."""

    start_line: int
    """Line number (1-based) where the snippet's first code line appears."""

    code: str
    """The code content of the snippet."""

    skipped_reason: Optional[str] = None
    forced_run: bool = False
    forced_concept: bool = False
    requires_files: list[str] | None = None  # paths relative to repo root


def find_markdown_files(paths: Iterable[str]) -> list[str]:
    """Return sorted absolute paths to Markdown/MDX files under the provided targets."""

    files: list[str] = []
    for base in paths:
        if not os.path.exists(base):
            continue
        if os.path.isfile(base):
            _, ext = os.path.splitext(base)
            if ext in SUPPORTED_EXTENSIONS:
                files.append(os.path.abspath(base))
            continue
        for root, _dirs, filenames in os.walk(base):
            for name in filenames:
                _, ext = os.path.splitext(name)
                if ext in SUPPORTED_EXTENSIONS:
                    files.append(os.path.abspath(os.path.join(root, name)))
    return sorted(files)


def extract_python_snippets(file_path: str, repo_root: str) -> list[Snippet]:
    """Extract runnable Python snippets from a Markdown/MDX file."""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    snippets: list[Snippet] = []
    snippet_index = 0

    def is_python_language_tag(tag: str) -> bool:
        tag = tag.strip().lower()
        if not tag:
            return False
        # handle cases like "python", "python title=...", "py"
        lang = tag.split()[0]
        return lang in {"python", "py"}

    i = 0
    while i < len(lines):
        start_match = FENCE_START_RE.match(lines[i])
        if not start_match:
            i += 1
            continue

        tag = (start_match.group("lang") or "").strip()
        if not is_python_language_tag(tag):
            i += 1
            continue

        snippet_index += 1
        line_no = i + 1

        markers: list[str] = []
        j = i - 1
        while j >= 0:
            prev = lines[j].strip()
            if prev == "":
                j -= 1
                continue
            if prev.startswith("<!--") and prev.endswith("-->"):
                markers.append(prev)
                j -= 1
                continue
            break

        pending_skipped_reason: Optional[str] = None
        pending_forced_run = False
        pending_forced_concept = False
        pending_requires_files: list[str] = []

        if TEST_IGNORE_MARK in markers:
            pending_skipped_reason = "test-ignore marker"
        if TEST_CONCEPT_MARK in markers:
            pending_forced_concept = True
        if TEST_RUN_MARK in markers:
            pending_forced_run = True
        for marker in markers:
            if marker.startswith(TEST_REQUIRE_FILES_PREFIX) and marker.endswith("-->"):
                content = marker[len(TEST_REQUIRE_FILES_PREFIX) : -3].strip()
                if content:
                    pending_requires_files.extend(content.split())

        block_lines: list[str] = []
        i += 1
        while i < len(lines) and not FENCE_END_RE.match(lines[i]):
            block_lines.append(lines[i])
            i += 1

        snippet = Snippet(
            file_path=file_path,
            relative_path=os.path.relpath(file_path, repo_root),
            snippet_index=snippet_index,
            start_line=line_no + 1,
            code="\n".join(block_lines).rstrip("\n"),
            skipped_reason=pending_skipped_reason,
            forced_run=pending_forced_run,
            forced_concept=pending_forced_concept,
            requires_files=pending_requires_files.copy() if pending_requires_files else None,
        )
        snippets.append(snippet)

        i += 1  # Skip closing fence

    return snippets


def _should_skip_snippet(snippet: Snippet, repo_root: str, skip_unsafe: bool) -> ExecutionResult | None:
    """Return an ExecutionResult for skipped snippets, or None if runnable."""

    if snippet.skipped_reason:
        return ExecutionResult(snippet=snippet, status=ExecutionStatus.SKIPPED, reason=snippet.skipped_reason)

    if snippet.forced_concept and not snippet.forced_run:
        return ExecutionResult(snippet=snippet, status=ExecutionStatus.SKIPPED, reason="concept marker")

    if snippet.requires_files:
        missing = [p for p in snippet.requires_files if not os.path.exists(os.path.join(repo_root, p))]
        if missing:
            return ExecutionResult(
                snippet=snippet, status=ExecutionStatus.SKIPPED, reason=f"missing required files: {', '.join(missing)}"
            )

    runnable = is_heuristically_runnable(snippet.code)
    if not runnable and not snippet.forced_run:
        return ExecutionResult(
            snippet=snippet, status=ExecutionStatus.SKIPPED, reason="heuristic: no imports (concept)"
        )

    if skip_unsafe:
        unsafe = contains_unsafe_pattern(snippet.code)
        if unsafe:
            return ExecutionResult(snippet=snippet, status=ExecutionStatus.SKIPPED, reason=f"unsafe pattern: {unsafe}")

    return None


def contains_unsafe_pattern(code: str) -> Optional[str]:
    """Return the unsafe pattern found in code, if any."""

    for pat in UNSAFE_PATTERNS:
        if pat.search(code):
            return pat.pattern
    return None


IMPORT_RE = re.compile(r"^\s*(?:from\s+\S+\s+import\s+|import\s+\S+)")


def is_heuristically_runnable(code: str) -> bool:
    """Heuristic to detect import statements signalling runnable code."""

    return any(IMPORT_RE.search(line) for line in code.splitlines())


class ExecutionStatus(Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ExecutionResult:
    snippet: Snippet
    status: ExecutionStatus
    return_code: Optional[int] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    reason: Optional[str] = None


def run_snippet(snippet: Snippet, timeout_seconds: int, cwd: str, skip_unsafe: bool) -> ExecutionResult:
    """Execute a single snippet and return the outcome."""
    skip_result = _should_skip_snippet(snippet, cwd, skip_unsafe)
    if skip_result is not None:
        return skip_result

    # Write to a temp file for better tracebacks (with file path and correct line numbers)
    # Use a stable informative temp file name in a dedicated temp dir
    safe_rel = snippet.relative_path.replace(os.sep, "__")
    temp_dir = os.path.join(tempfile.gettempdir(), "doc_snippet_tests")
    os.makedirs(temp_dir, exist_ok=True)
    temp_name = f"{safe_rel}__snippet_{snippet.snippet_index}.py"
    temp_path = os.path.join(temp_dir, temp_name)

    # Prepend a line directive comment to facilitate mapping if needed
    prelude = textwrap.dedent(
        f"""
        # File: {snippet.relative_path}
        # Snippet: {snippet.snippet_index}
        # Start line in source: {snippet.start_line}
        """
    ).lstrip("\n")

    with open(temp_path, "w", encoding="utf-8") as tf:
        tf.write(prelude)
        tf.write(snippet.code)
        tf.write("\n")

    try:
        completed = subprocess.run(
            [sys.executable, temp_path],
            check=False,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            text=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        if completed.returncode == 0:
            return ExecutionResult(
                snippet=snippet, status=ExecutionStatus.PASSED, return_code=0, stdout=completed.stdout
            )

        return ExecutionResult(
            snippet=snippet,
            status=ExecutionStatus.FAILED,
            return_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
    except subprocess.TimeoutExpired as exc:
        return ExecutionResult(
            snippet=snippet,
            status=ExecutionStatus.FAILED,
            reason=f"timeout after {timeout_seconds}s",
            stdout=exc.stdout or None,
            stderr=(exc.stderr or "") + f"\n[timeout after {timeout_seconds}s]",
        )


def print_failure_annotation(result: ExecutionResult) -> None:
    """Print a GitHub Actions error annotation so failures are clickable in CI logs."""
    rel = result.snippet.relative_path
    line = result.snippet.start_line
    # Escape newlines and percents per GH annotation rules
    message = f"Doc snippet #{result.snippet.snippet_index} failed"
    stderr_text = result.stderr.strip() if result.stderr else ""
    stdout_text = result.stdout.strip() if result.stdout else ""
    details = stderr_text or stdout_text
    if result.reason:
        details = f"{result.reason}\n\n" + details
    details = details.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    sys.stdout.write(f"::error file={rel},line={line}::{message} — see details below%0A{details}\n")


def process_file_snippets(
    file_rel: str, snippets: list[Snippet], repo_root: str, timeout_seconds: int, allow_unsafe: bool, verbose: bool
) -> tuple[list[ExecutionResult], dict[str, int]]:
    """Process all snippets in a single markdown file and return results and statistics."""
    if verbose:
        print(f"[RUN] {file_rel}")
    else:
        print(f"Running {file_rel} ({len(snippets)} snippet(s))")

    results: list[ExecutionResult] = []
    file_passed = file_failed = file_skipped = 0

    for snippet in snippets:
        result = run_snippet(snippet, timeout_seconds=timeout_seconds, cwd=repo_root, skip_unsafe=not allow_unsafe)
        results.append(result)

        if result.status == ExecutionStatus.PASSED:
            file_passed += 1
            if verbose:
                print(f"[PASS] {snippet.relative_path}#snippet{snippet.snippet_index} (line {snippet.start_line})")
        elif result.status == ExecutionStatus.SKIPPED:
            file_skipped += 1
            if verbose:
                reason = f" — {result.reason}" if result.reason else ""
                print(f"[SKIP] {snippet.relative_path}#snippet{snippet.snippet_index}{reason}")
        else:
            file_failed += 1
            print_failure_annotation(result)
            # Also print a concise human-readable failure line
            print(
                f"FAILED {snippet.relative_path}:snippet{snippet.snippet_index} "
                f"(line {snippet.start_line}) — rc={result.return_code or 'N/A'}"
            )
            if result.stdout and result.stdout.strip():
                print("--- stdout ---\n" + result.stdout)
            if result.stderr and result.stderr.strip():
                print("--- stderr ---\n" + result.stderr)
    stats = {"total": len(snippets), "passed": file_passed, "failed": file_failed, "skipped": file_skipped}

    return results, stats


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point for snippet execution."""
    parser = argparse.ArgumentParser(description="Test Python code snippets in Docusaurus docs")
    parser.add_argument(
        "targets",
        nargs="*",
        help=("Optional positional list of files or directories to scan. If omitted, --paths is used."),
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        default=["docs", "versioned_docs"],
        help=(
            "Fallback directories or files to scan when no positional targets are provided "
            "(defaults to docs and versioned_docs)"
        ),
    )
    parser.add_argument("--timeout-seconds", type=int, default=30, help="Timeout per snippet execution (seconds)")
    parser.add_argument(
        "--allow-unsafe", action="store_true", help="Allow execution of snippets with potentially unsafe patterns"
    )
    parser.add_argument("--verbose", action="store_true", help="Print verbose logs")

    args = parser.parse_args(argv)
    repo_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    raw_paths = args.targets if args.targets else args.paths
    scan_paths = [os.path.join(repo_root, p) if not os.path.isabs(p) else p for p in raw_paths]

    md_files = find_markdown_files(scan_paths)
    if args.verbose:
        print(f"Repo root: {repo_root}")
        print(f"Scanning targets: {', '.join(raw_paths)}")
        print(f"Discovered {len(md_files)} Markdown files")
    else:
        print(f"Discovered {len(md_files)} Markdown files")

    all_snippets: list[Snippet] = []
    for idx, fpath in enumerate(md_files, start=1):
        rel = os.path.relpath(fpath, repo_root)
        if args.verbose:
            print(f"[SCAN {idx}/{len(md_files)}] {rel}")
        snippets = extract_python_snippets(fpath, repo_root)
        if snippets:
            all_snippets.extend(snippets)
        if args.verbose:
            print(f"[FOUND] {rel}: {len(snippets)} python snippet(s)")

    if args.verbose:
        print(f"Extracted {len(all_snippets)} Python snippets")
    else:
        print(f"Total Python snippets found: {len(all_snippets)}")

    total = len(all_snippets)
    passed = 0
    failed = 0
    skipped = 0
    results: list[ExecutionResult] = []

    # Ensure deterministic execution order grouped by file, then line
    all_snippets.sort(key=lambda s: (s.relative_path, s.start_line, s.snippet_index))

    # Group by file
    file_to_snippets: dict[str, list[Snippet]] = {}
    for sn in all_snippets:
        file_to_snippets.setdefault(sn.relative_path, []).append(sn)

    file_stats: dict[str, dict[str, int]] = {}
    for file_rel, snippets in file_to_snippets.items():
        file_results, stats = process_file_snippets(
            file_rel=file_rel,
            snippets=snippets,
            repo_root=repo_root,
            timeout_seconds=args.timeout_seconds,
            allow_unsafe=args.allow_unsafe,
            verbose=args.verbose,
        )
        results.extend(file_results)
        file_stats[file_rel] = stats

        # Update totals
        passed += stats["passed"]
        failed += stats["failed"]
        skipped += stats["skipped"]

    print(f"Summary: total={total}, passed={passed}, failed={failed}, skipped={skipped}")

    # Per-file summary - show only files with failures by default, all in verbose mode
    print("Files summary:")
    for file_rel in sorted(file_stats.keys()):
        fs = file_stats[file_rel]
        # Show file if it has failures or if verbose mode is on
        if fs["failed"] > 0 or args.verbose:
            print(
                f" - {file_rel}: total={fs['total']}, passed={fs['passed']}, "
                f"failed={fs['failed']}, skipped={fs['skipped']}"
            )

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
