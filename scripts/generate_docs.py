#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys


def main() -> int:
    """Run the API doc generator and clean up escaped JSX comment syntax."""
    # 1. Run the original doc generation
    cmd = ["haystack-pydoc", "pydoc", "tmp_api_reference"]
    print(f"Running: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    print(res.stdout)
    if res.stderr:
        print(res.stderr, file=sys.stderr)
    if res.returncode != 0:
        return res.returncode

    # 2. Post-process files in tmp_api_reference
    tmp_dir = "tmp_api_reference"
    if not os.path.exists(tmp_dir):
        print(f"Error: {tmp_dir} does not exist.")
        return 1

    replacements = {"{/\\*": "{/*", "\\*/}": "*/}"}

    count = 0
    for root, _, files in os.walk(tmp_dir):
        for file in files:
            if file.endswith((".md", ".mdx")):
                filepath = os.path.join(root, file)
                with open(filepath, encoding="utf-8") as f:
                    content = f.read()

                new_content = content
                for old, new in replacements.items():
                    new_content = new_content.replace(old, new)

                if new_content != content:
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(new_content)
                    count += 1

    print(f"Post-processed {count} generated reference files to fix JSX comments.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
