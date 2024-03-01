import ast
import os
from _ast import Import, ImportFrom
from pathlib import Path

import isort
import toml
from pyproject_parser import PyProject


def test_for_missing_dependencies() -> None:
    # All tools out there are too powerful because they find all the imports in the haystack package
    # We need to find only the top level imports

    #### Collect imports
    top_level_imports = set()
    for path in Path("haystack").glob("**/*.py"):
        content = path.read_text(encoding="utf-8")
        tree = ast.parse(content)
        for item in tree.body:
            if isinstance(item, Import):
                module = item.names[0].name
            elif isinstance(item, ImportFrom) and item.level == 0:  # level > 1 are relative imports
                module = item.module
            else:
                # we only care about imports
                break

            top_level_imports.add(module.split(".")[0])

    custom_std_libs = {"yaml"}
    third_party_modules = {
        module
        for module in top_level_imports
        if isort.place_module(module) == "THIRDPARTY" and module not in custom_std_libs
    }

    #### Load specified dependencies
    parsed = toml.load("pyproject.toml")
    # Pyproject complains about our pyproject.toml file, so we need to parse only the dependencies
    # We still need `PyProject` to parse the dependencies (think of markers and stuff)
    only_dependencies = {"project": {"name": "test", "dependencies": parsed["project"]["dependencies"]}}
    project_dependencies = PyProject.project_table_parser.parse(only_dependencies["project"], set_defaults=True)[
        "dependencies"
    ]

    library_module_mapper = {"python-dateutil": "dateutil"}

    project_dependency_modules = set()
    for dep in project_dependencies:
        if dep.name in library_module_mapper:
            project_dependency_modules.add(library_module_mapper[dep.name])

        project_dependency_modules.add(dep.name.replace("-", "_"))

    #### And now finally; the check
    for module in third_party_modules:
        assert module in project_dependency_modules, f"Module {module} is not in the dependencies"
