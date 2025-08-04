import importlib
import os
import sys
import traceback
from pathlib import Path
from typing import List, Optional

from haystack import logging  # pylint: disable=unused-import  # this is needed to avoid circular imports


def validate_module_imports(root_dir: str, exclude_subdirs: Optional[list[str]] = None) -> tuple[list, list]:
    """
    Recursively search for all Python modules and attempt to import them.

    This includes both packages (directories with __init__.py) and individual Python files.
    """
    imported = []
    failed = []
    exclude_subdirs = (exclude_subdirs or []) + ["__pycache__"]

    # Add the root directory to the Python path
    sys.path.insert(0, root_dir)
    base_path = Path(root_dir)

    for root, _, files in os.walk(root_dir):
        if any(subdir in root for subdir in exclude_subdirs):
            continue

        # Convert path to module format
        module_path = ".".join(Path(root).relative_to(base_path.parent).parts)
        python_files = [f for f in files if f.endswith(".py")]

        # Try importing package and individual files
        for file in python_files:
            try:
                if file == "__init__.py":
                    module_to_import = module_path
                else:
                    module_name = os.path.splitext(file)[0]
                    module_to_import = f"{module_path}.{module_name}" if module_path else module_name

                importlib.import_module(module_to_import)
                imported.append(module_to_import)
            except:
                failed.append({"module": module_to_import, "traceback": traceback.format_exc()})

    return imported, failed


def main():
    """
    This script checks that all Haystack modules can be imported successfully.

    This includes both packages and individual Python files.
    This can detect several issues, such as:
    - Syntax errors in Python files
    - Missing dependencies
    - Circular imports
    - Incorrect type hints without forward references
    """
    # Add any subdirectories you want to skip during import checks ("__pycache__" is skipped by default)
    exclude_subdirs = ["testing"]

    print("Checking imports from all Haystack modules...")
    imported, failed = validate_module_imports(root_dir="haystack", exclude_subdirs=exclude_subdirs)

    if not imported:
        print("\nNO MODULES WERE IMPORTED")
        sys.exit(1)

    print(f"\nSUCCESSFULLY IMPORTED {len(imported)} MODULES")

    if failed:
        print(f"\nFAILED TO IMPORT {len(failed)} MODULES:")
        for fail in failed:
            print(f"  - {fail['module']}")

        print("\nERRORS:")
        for fail in failed:
            print(f"  - {fail['module']}\n")
            print(f"    {fail['traceback']}\n\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
