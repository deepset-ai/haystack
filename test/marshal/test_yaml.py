# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import pytest

from haystack.marshal.yaml import YamlMarshaller


@pytest.fixture
def yaml_data():
    return {"key": "value", 1: 0.221, "list": [1, 2, 3], "tuple": (1, None, True), "dict": {"set": {False}}}


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


def test_yaml_unmarshal(yaml_data, serialized_yaml_str):
    marshaller = YamlMarshaller()
    unmarshalled = marshaller.unmarshal(serialized_yaml_str)
    assert unmarshalled == yaml_data
