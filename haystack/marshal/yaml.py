# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import yaml


# Custom YAML safe loader that supports loading Python tuples
class YamlLoader(yaml.SafeLoader):
    def construct_python_tuple(self, node: yaml.SequenceNode) -> tuple:
        """Construct a Python tuple from the sequence."""
        return tuple(self.construct_sequence(node))


class YamlDumper(yaml.SafeDumper):
    def represent_tuple(self, data: tuple) -> yaml.SequenceNode:
        """Represent a Python tuple."""
        return self.represent_sequence("tag:yaml.org,2002:python/tuple", data)

    def represent_str(self, data: str) -> yaml.ScalarNode:
        """Represent a string, using single-quoted style for strings containing backslashes.

        This ensures that backslash sequences (e.g. ``\\b``, ``\\w``) in regex
        patterns and file paths are preserved literally during YAML round-tripping.
        Without this, a plain scalar like ``remove_regex: \\b\\w+\\b`` may be
        misinterpreted on some YAML / Python versions, causing
        ``ReaderError`` or ``SyntaxWarning`` on load (#11093).
        """
        if "\\" in data:
            return self.represent_scalar(
                "tag:yaml.org,2002:str", data, style="'"
            )
        return self.represent_scalar("tag:yaml.org,2002:str", data)


YamlDumper.add_representer(tuple, YamlDumper.represent_tuple)
YamlDumper.add_representer(str, YamlDumper.represent_str)
YamlLoader.add_constructor("tag:yaml.org,2002:python/tuple", YamlLoader.construct_python_tuple)


class YamlMarshaller:
    def marshal(self, dict_: dict[str, Any]) -> str:
        """Return a YAML representation of the given dictionary."""
        try:
            return yaml.dump(dict_, Dumper=YamlDumper)
        except yaml.representer.RepresenterError as e:
            raise TypeError(
                "Error dumping pipeline to YAML - Ensure that all pipeline components only serialize basic Python types"
            ) from e

    def unmarshal(self, data_: str | bytes | bytearray) -> dict[str, Any]:
        """Return a dictionary from the given YAML data."""
        try:
            return yaml.load(data_, Loader=YamlLoader)
        except yaml.constructor.ConstructorError as e:
            raise TypeError(
                "Error loading pipeline from YAML - Ensure that all pipeline "
                "components only serialize basic Python types"
            ) from e
