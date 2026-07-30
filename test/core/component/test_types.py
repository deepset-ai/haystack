# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from pandas import DataFrame

from haystack.core.component.types import InputSocket


def test_input_socket_with_a_default_that_cannot_be_compared_is_not_mandatory():
    """`is_mandatory` uses `is`, because comparing a DataFrame with `==` returns a DataFrame, not True or False."""
    socket = InputSocket(name="value", type=DataFrame, default_value=DataFrame.from_dict([{"value": 42}]))

    assert socket.is_mandatory is False
