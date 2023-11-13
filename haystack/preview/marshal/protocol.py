from typing import Protocol, Dict, Any, Union


class Marshaller(Protocol):
    def marshal(self, dict_: Dict[str, Any]) -> str:
        ...

    def unmarshal(self, data_: Union[str, bytes, bytearray]) -> Dict[str, Any]:
        ...
