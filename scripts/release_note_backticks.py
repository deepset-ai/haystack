# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Pre-commit hook that enforces reStructuredText inline-code syntax in reno release notes.

Release notes under ``releasenotes/notes/*.yaml`` are rendered as reStructuredText, where inline
code must use double backticks (``like_this``). A common mistake is to use Markdown-style single
backticks (`like_this`), which RST renders as italic "interpreted text" instead of code.

By default the hook rewrites single backticks to double backticks and exits non-zero when it
changes a file, so the rewrite can be reviewed before re-staging; ``--check`` only reports.
It leaves existing double backticks ``code`` untouched and skips RST roles (:func:`x`) and
hyperlink references (`text <url>`_).
"""

import argparse
import re
import sys

# A single-backtick inline-code span that should use double backticks. The look-arounds skip
# valid RST: ``...`` pairs (the backtick-adjacent guards), roles like :func:`x` (no ':' before
# the opening backtick), and hyperlink refs like `text <url>`_ (no '_' after the closing one).
SINGLE_BACKTICK_RE = re.compile(r"(?<![`:])`(?!`)([^`\n]+?)`(?![`_])")


def fix_text(text: str) -> str:
    """Return ``text`` with single-backtick inline code rewritten to double backticks."""
    return SINGLE_BACKTICK_RE.sub(r"``\1``", text)


def main() -> int:
    """Entry point for the pre-commit hook."""
    parser = argparse.ArgumentParser(description="Enforce double backticks in reno release notes.")
    parser.add_argument("--check", action="store_true", help="Report problems without modifying files.")
    parser.add_argument("files", nargs="*")
    args = parser.parse_args()

    ret = 0
    for path in args.files:
        with open(path, encoding="utf-8") as f:
            original = f.read()
        fixed = fix_text(original)
        if fixed == original:
            continue
        ret = 1
        print(f"{'single backtick in' if args.check else 'fixed'}: {path}", file=sys.stderr)
        if not args.check:
            with open(path, "w", encoding="utf-8") as f:
                f.write(fixed)

    if ret:
        hint = "Run without --check to fix automatically." if args.check else "Review and re-stage the changes."
        print(f"\nrelease notes need double backticks (``like_this``) for inline code. {hint}", file=sys.stderr)
    return ret


if __name__ == "__main__":
    raise SystemExit(main())
