import importlib


def load_class(full_class_path: str):
    """
    Load a class from a string representation of its path e.g. "module.submodule.class_name"
    """
    module_path, _, class_name = full_class_path.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
