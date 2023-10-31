from typing import Dict, Any, Union

import yaml


class YamlMarshaller:
    def marshal(self, dict_: Dict[str, Any]) -> str:
        return yaml.dump(dict_)

    def unmarshal(self, data_: Union[str, bytes, bytearray]) -> Dict[str, Any]:
        return yaml.safe_load(data_)
