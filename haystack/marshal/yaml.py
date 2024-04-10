from typing import Any, Dict, Union

import yaml


class YamlMarshaller:
    def marshal(self, dict_: Dict[str, Any]) -> str:
        """Return a YAML representation of the given dictionary."""
        return yaml.dump(dict_)

    def unmarshal(self, data_: Union[str, bytes, bytearray]) -> Dict[str, Any]:
        """Return a dictionary from the given YAML data."""
        return yaml.safe_load(data_)
