import argparse
import re
import sys
from pathlib import Path

import toml

matcher = re.compile(r"farm-haystack\[(.+)\]")
parser = argparse.ArgumentParser(
    prog="pyproject_to_requirements.py", description="Convert pyproject.toml to requirements.txt"
)
parser.add_argument("pyproject_path")
parser.add_argument("--extra", default="")


def resolve(target: str, extras: dict, results: set):
    if target not in extras:
        results.add(target)
        return

    for t in extras[target]:
        m = matcher.match(t)
        if m:
            for i in m.group(1).split(","):
                resolve(i, extras, results)
        else:
            resolve(t, extras, results)


def main(pyproject_path: Path, extra: str = ""):
    content = toml.load(pyproject_path)
    # basic set of dependencies
    deps = set(content["project"]["dependencies"])

    if extra:
        extras = content["project"]["optional-dependencies"]
        resolve(extra, extras, deps)

    sys.stdout.write("\n".join(sorted(deps)))
    sys.stdout.write("\n")


if __name__ == "__main__":
    args = parser.parse_args()
    pyproject_path = Path(args.pyproject_path).absolute()

    main(pyproject_path, args.extra)
