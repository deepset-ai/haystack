# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Union

import yaml


# Custom YAML safe loader that supports loading Python tuples
class YamlLoader(yaml.SafeLoader):  # pylint: disable=too-many-ancestors
    def construct_python_tuple(self, node: yaml.SequenceNode):
        """Construct a Python tuple from the sequence."""
        return tuple(self.construct_sequence(node))


YamlLoader.add_constructor("tag:yaml.org,2002:python/tuple", YamlLoader.construct_python_tuple)


class YamlMarshaller:
    def marshal(self, dict_: Dict[str, Any]) -> str:
        """Return a YAML representation of the given dictionary."""
        return yaml.dump(dict_)

    def unmarshal(self, data_: Union[str, bytes, bytearray]) -> Dict[str, Any]:
        """Return a dictionary from the given YAML data."""
        return yaml.load(data_, Loader=YamlLoader)
