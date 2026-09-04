# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack.testing.sample_components import Double


def test_double_default():
    component = Double()
    results = component.run(value=10)
    assert results == {"value": 20}
