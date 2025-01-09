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


class YamlDumper(yaml.SafeDumper):  # pylint: disable=too-many-ancestors
    def represent_tuple(self, data: tuple):
        """Represent a Python tuple."""
        return self.represent_sequence("tag:yaml.org,2002:python/tuple", data)


YamlDumper.add_representer(tuple, YamlDumper.represent_tuple)
YamlLoader.add_constructor("tag:yaml.org,2002:python/tuple", YamlLoader.construct_python_tuple)


class YamlMarshaller:
    def marshal(self, dict_: Dict[str, Any]) -> str:
        """Return a YAML representation of the given dictionary."""
        try:
            return yaml.dump(dict_, Dumper=YamlDumper)
        except yaml.representer.RepresenterError as e:
            raise TypeError(
                "Error dumping pipeline to YAML - Ensure that all pipeline components only serialize basic Python types"
            ) from e

    def unmarshal(self, data_: Union[str, bytes, bytearray]) -> Dict[str, Any]:
        """Return a dictionary from the given YAML data."""
        try:
            return yaml.load(data_, Loader=YamlLoader)
        except yaml.constructor.ConstructorError as e:
            raise TypeError(
                "Error loading pipeline from YAML - Ensure that all pipeline "
                "components only serialize basic Python types"
            ) from e
