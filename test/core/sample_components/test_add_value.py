# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack.testing.sample_components import AddFixedValue


def test_run():
    component = AddFixedValue()
    results = component.run(value=50, add=10)
    assert results == {"result": 60}
