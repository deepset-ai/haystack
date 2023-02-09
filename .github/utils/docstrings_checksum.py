from pathlib import Path
from typing import Generator

import ast
import hashlib


def docstrings_checksum(python_files: Generator[Path, None, None]):
    files_content = (f.read_text() for f in python_files)
    trees = (ast.parse(c) for c in files_content)

    # Get all docstrings from async functions, functions,
    # classes and modules definitions
    docstrings = []
    for tree in trees:
        for node in ast.walk(tree):
            if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef, ast.ClassDef, ast.Module)):
                # Skip all node types that can't have docstrings to prevent failures
                continue
            docstring = ast.get_docstring(node)
            if docstring:
                docstrings.append(docstring)

    # Sort them to be safe, since ast.walk() returns
    # nodes in no specified order.
    # See https://docs.python.org/3/library/ast.html#ast.walk
    docstrings.sort()

    return hashlib.md5(str(docstrings).encode("utf-8")).hexdigest()


if __name__ == "__main__":
    # Get all Haystack and rest_api python files
    root = Path(".").absolute().parent.parent
    python_files = root.glob("+(haystack|rest_api)/**/*.py")

    md5 = docstrings_checksum(python_files)
    print(md5)
