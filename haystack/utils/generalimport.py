import sys

from generalimport import FakeModule, MissingOptionalDependency


# TODO: remove this function once this PR is merged and released by generalimport:
# https://github.com/ManderaGeneral/generalimport/pull/25
def is_imported(module_name: str) -> bool:
    """
    Returns True if the module was actually imported, False, if generalimport mocked it.
    """
    module = sys.modules.get(module_name)
    try:
        return bool(module) and not isinstance(module, FakeModule)
    except MissingOptionalDependency:
        # isinstance() raises MissingOptionalDependency: fake module
        pass
    return False
