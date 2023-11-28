# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from haystack.testing.sample_components import Subtract
from haystack.core.serialization import component_to_dict, component_from_dict


def test_subtract():
    component = Subtract()
    results = component.run(first_value=10, second_value=7)
    assert results == {"difference": 3}
