# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack.marshal.yaml import YamlMarshaller


class InvalidClass:
    def __init__(self) -> None:
        self.a = 1
        self.b = None
        self.c = "string"


@pytest.fixture
def yaml_data():
    return {"key": "value", 1: 0.221, "list": [1, 2, 3], "tuple": (1, None, True), "dict": {"set": {False}}}


@pytest.fixture
def invalid_yaml_data():
    return {"key": "value", 1: 0.221, "list": [1, 2, 3], "tuple": (1, InvalidClass(), True), "dict": {"set": {False}}}


@pytest.fixture
def serialized_yaml_str():
    return """key: value
1: 0.221
list:
- 1
- 2
- 3
tuple: !!python/tuple
- 1
- null
- true
dict:
  set: !!set
    false: null
"""


def test_yaml_marshal(yaml_data, serialized_yaml_str):
    marshaller = YamlMarshaller()
    marshalled = marshaller.marshal(yaml_data)
    assert isinstance(marshalled, str)
    assert marshalled.strip().replace("\n", "") == serialized_yaml_str.strip().replace("\n", "")


def test_yaml_marshal_invalid_type(invalid_yaml_data):
    with pytest.raises(TypeError, match="basic Python types"):
        marshaller = YamlMarshaller()
        _ = marshaller.marshal(invalid_yaml_data)


def test_yaml_unmarshal(yaml_data, serialized_yaml_str):
    marshaller = YamlMarshaller()
    unmarshalled = marshaller.unmarshal(serialized_yaml_str)
    assert unmarshalled == yaml_data


class TestYamlBackslashRoundtrip:
    """Strings with backslashes must round-trip correctly through YAML.

    Regression test for #11093: regex patterns and file paths containing
    backslash sequences must survive YAML round-tripping.
    """

    def test_single_backslash_sequence(self):
        m = YamlMarshaller()
        data = {"regex": r"\b\w+\b"}
        assert m.unmarshal(m.marshal(data)) == data

    def test_complex_regex(self):
        m = YamlMarshaller()
        data = {"regex": r"(?u)\b\w+\b"}
        assert m.unmarshal(m.marshal(data)) == data

    def test_windows_path(self):
        m = YamlMarshaller()
        data = {"path": r"C:\Users\test\file.txt"}
        assert m.unmarshal(m.marshal(data)) == data

    def test_no_backslash_unchanged(self):
        m = YamlMarshaller()
        data = {"text": "hello world"}
        dumped = m.marshal(data)
        # Plain scalars should not be quoted when no backslash is present
        assert "hello world" in dumped
        assert m.unmarshal(dumped) == data

    def test_backslash_string_is_single_quoted(self):
        m = YamlMarshaller()
        data = {"regex": r"\s+"}
        dumped = m.marshal(data)
        # Single-quoted scalars use ' as delimiter to prevent escape interpretation
        assert "'\\s+'" in dumped or "'\\\\s+'" in dumped

    def test_pipe_roundtrip_with_regex(self):
        """Full Pipeline round-trip with components that use regex parameters."""
        from haystack import Pipeline
        from haystack.components.preprocessors import DocumentCleaner

        pipe = Pipeline()
        cleaner = DocumentCleaner(remove_regex=r"\b\w+\b", replace_regexes={r"\n\n+": "\n"})
        pipe.add_component("cleaner", cleaner)

        yaml_str = pipe.dumps()
        loaded = Pipeline.loads(yaml_str)

        # Check regex was preserved
        loaded_cleaner = loaded.graph.nodes["cleaner"]["instance"]
        assert loaded_cleaner.remove_regex == r"\b\w+\b"
        assert r"\n\n+" in loaded_cleaner.replace_regexes
