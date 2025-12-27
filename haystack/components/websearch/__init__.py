import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

__all__ = ["SerperDevWebSearch", "SearchApiWebSearch", "BrightDataWebSearch"]

_import_structure = {
    "searchapi": ["SearchApiWebSearch"],
    "serper_dev": ["SerperDevWebSearch"],
    "brightdata": ["BrightDataWebSearch"],
}

if TYPE_CHECKING:
    from .brightdata import BrightDataWebSearch as BrightDataWebSearch
    from .searchapi import SearchApiWebSearch as SearchApiWebSearch
    from .serper_dev import SerperDevWebSearch as SerperDevWebSearch
else:
    sys.modules[__name__] = LazyImporter(name=__name__, module_file=__file__, import_structure=_import_structure)
