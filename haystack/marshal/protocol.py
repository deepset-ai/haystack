# pylint: disable=unnecessary-ellipsis
from typing import Any, Dict, Protocol, Union


class Marshaller(Protocol):
    def marshal(self, dict_: Dict[str, Any]) -> str:
        "Convert a dictionary to its string representation"
        ...

    def unmarshal(self, data_: Union[str, bytes, bytearray]) -> Dict[str, Any]:
        """Convert a marshalled object to its dictionary representation"""
        ...
