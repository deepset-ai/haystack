# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack.core.serialization import component_from_dict, component_to_dict
from haystack.components.joiners.string_joiner import StringJoiner


class TestStringJoiner:
    def test_init(self):
        joiner = StringJoiner()
        assert isinstance(joiner, StringJoiner)

    def test_to_dict(self):
        joiner = StringJoiner()
        data = component_to_dict(joiner, name="string_joiner")
        assert data == {"type": "haystack.components.joiners.string_joiner.StringJoiner", "init_parameters": {}}

    def test_from_dict(self):
        data = {"type": "haystack.components.joiners.string_joiner.StringJoiner", "init_parameters": {}}
        string_joiner = component_from_dict(StringJoiner, data=data, name="string_joiner")
        assert isinstance(string_joiner, StringJoiner)

    def test_empty_list(self):
        joiner = StringJoiner()
        result = joiner.run([])
        assert result == {"strings": []}

    def test_single_string(self):
        joiner = StringJoiner()
        result = joiner.run("a")
        assert result == {"strings": ["a"]}

    def test_two_strings(self):
        joiner = StringJoiner()
        result = joiner.run(["a", "b"])
        assert result == {"strings": ["a", "b"]}
