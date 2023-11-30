# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from haystack.testing.sample_components import Parity
from haystack.core.serialization import component_to_dict, component_from_dict


def test_parity():
    component = Parity()
    results = component.run(value=1)
    assert results == {"odd": 1}
    results = component.run(value=2)
    assert results == {"even": 2}
