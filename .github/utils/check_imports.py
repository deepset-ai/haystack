import os
import sys
import importlib
import traceback
from haystack import logging  # this is needed to avoid circular imports

"""
This script checks that all Haystack packages can be imported successfully.
This can detect several issues.
One example is forgetting to use a forward reference for a type hint coming from a lazy import.
"""

def import_packages_with_init(directory):
    """
    Recursively search for directories with __init__.py and import them safely.
    """
    imported = []
    failed = []

    sys.path.insert(0, directory)

    for root, _, files in os.walk(directory):
        # skip directories without __init__.py
        if not '__init__.py' in files:
            continue

        init_path = os.path.join(root, '__init__.py')

        # skip haystack/__init__.py to avoid circular imports
        if init_path.endswith(f'{directory}/__init__.py'):
            continue

        relative_path = os.path.relpath(root, directory)
        module_name = relative_path.replace(os.path.sep, '.')
        if module_name == '.':
            module_name = os.path.basename(directory)

        try:
            importlib.import_module(module_name)
            imported.append(module_name)
        except:
            failed.append({
                'module': module_name,
                'traceback': traceback.format_exc()
                })

    return imported, failed


if __name__ == '__main__':
    print("Checking imports from Haystack packages...")
    imported, failed = import_packages_with_init("haystack")

    print(f"\nSUCCESSFULLY IMPORTED {len(imported)} PACKAGES:")
    for package in sorted(imported):
        print(f"  - {package}")

    if failed:
        print(f"\nFAILED TO IMPORT {len(failed)} PACKAGES:")
        for fail in failed:
            print(f"  - {fail['module']}")

        print("\nERRORS:")
        for fail in failed:
            print(f"  - {fail['module']}\n")
            print(f"    {fail['traceback']}\n\n")
        sys.exit(1)
