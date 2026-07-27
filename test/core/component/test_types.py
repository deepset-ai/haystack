# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from pandas import DataFrame

from haystack.core.component.types import InputSocket


def test_input_socket_with_a_default_that_cannot_be_compared_is_not_mandatory():
    """
    `is_mandatory` tests for the sentinel by identity, so it never compares the default it was given.

    Comparing with '==' returns a DataFrame rather than a bool here, which makes `is_mandatory` unusable as a
    condition.
    """
    socket = InputSocket(name="value", type=DataFrame, default_value=DataFrame.from_dict([{"value": 42}]))

    assert socket.is_mandatory is False
