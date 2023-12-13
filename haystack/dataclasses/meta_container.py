from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any, Dict, Protocol

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

    class MetaContainer(Protocol, DataclassInstance):
        meta: Dict[str, Any]

else:
    MetaContainer = Any
