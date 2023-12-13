from __future__ import annotations
from dataclasses import Field

from typing import Any, Dict, Protocol, runtime_checkable, ClassVar


@runtime_checkable
class MetaContainer(Protocol):
    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]  # marks it as dataclass
    meta: Dict[str, Any]
