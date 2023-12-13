from typing import Any, Dict, Protocol, runtime_checkable
from dataclasses import dataclass


@runtime_checkable
@dataclass
class MetaContainer(Protocol):
    meta: Dict[str, Any]
