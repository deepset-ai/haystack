from typing import Optional, Any
import importlib
import logging


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
