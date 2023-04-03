from typing import Optional
import importlib
import logging


def optional_import(import_path: str, import_target: Optional[str], error_msg: str, importer_module: str):
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
            f"{import_path}{'.' if import_target else ''}{import_target} can't be imported. {error_msg} Error raised: {exc}"
        )
        return None
