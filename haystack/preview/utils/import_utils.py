from typing import Optional, Any

import sys
import importlib
import logging

from generalimport import FakeModule, MissingOptionalDependency


#
# TEMPORARY: remove once generalimport>0.3.1 is released
#
def is_imported(module_name: str) -> bool:
    """
    Returns True if the module was actually imported, False, if generalimport mocked it.
    """
    module = sys.modules.get(module_name)
    try:
        return bool(module and not isinstance(module, FakeModule))
    except MissingOptionalDependency as exc:
        # isinstance() raises MissingOptionalDependency: fake module
        pass
    return False


def optional_import(import_path: str, import_target: Optional[str], error_msg: str, importer_module: str) -> Any:
    """
    Imports an optional dependency. Emits a DEBUG log if the dependency is missing.
    """
    try:
        module = importlib.import_module(import_path)
        if import_target:
            return getattr(module, import_target)
        return module
    except ImportError as exc:
        logging.getLogger(importer_module).debug(
            "%s%s%s can't be imported: %s Error raised: %s",
            import_path,
            "." if import_target else "",
            import_target,
            error_msg,
            exc,
        )
        return None
