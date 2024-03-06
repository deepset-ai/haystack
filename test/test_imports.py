import ast
import os
from _ast import Import, ImportFrom
from pathlib import Path

import isort
import toml
from pyproject_parser import PyProject

# Some libraries have different names in the import and in the dependency
# If below test fails due to that, add the library name to the dictionary
LIBRARY_NAMES_TO_MODULE_NAMES = {"python-dateutil": "dateutil"}

# Some standard libraries are not detected by isort. If below test fails due to that, add the library name to the set.
ADDITIONAL_STD_LIBS = {"yaml"}


def test_for_missing_dependencies() -> None:
    # We implement this manual check because
    # - All tools out there are too powerful because they find all the imports in the haystack package
    # - if we import all modules to check the imports we don't find issues with direct dependencies which are also
    #   sub-dependencies of other dependencies

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

    third_party_modules = {
        module
        for module in top_level_imports
        if isort.place_module(module) == "THIRDPARTY" and module not in ADDITIONAL_STD_LIBS
    }

    #### Load specified dependencies
    parsed = toml.load("pyproject.toml")
    # Pyproject complains about our pyproject.toml file, so we need to parse only the dependencies
    # We still need `PyProject` to parse the dependencies (think of markers and stuff)
    only_dependencies = {"project": {"name": "test", "dependencies": parsed["project"]["dependencies"]}}
    project_dependencies = PyProject.project_table_parser.parse(only_dependencies["project"], set_defaults=True)[
        "dependencies"
    ]

    project_dependency_modules = set()
    for dep in project_dependencies:
        if dep.name in LIBRARY_NAMES_TO_MODULE_NAMES:
            project_dependency_modules.add(LIBRARY_NAMES_TO_MODULE_NAMES[dep.name])

        project_dependency_modules.add(dep.name.replace("-", "_"))

    #### And now finally; the check
    for module in third_party_modules:
        assert module in project_dependency_modules, f"Module {module} is not in the dependencies"
