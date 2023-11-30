# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from haystack.testing.sample_components import Repeat
from haystack.core.serialization import component_to_dict, component_from_dict


def test_repeat_default():
    component = Repeat(outputs=["one", "two"])
    results = component.run(value=10)
    assert results == {"one": 10, "two": 10}
