# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack.core.component.types import InputSocket


class _IncomparableValue:  # noqa: PLW1641  # __hash__ is irrelevant; this only needs a hostile __eq__
    """Stands in for values such as numpy arrays, whose `__eq__` does not return a bool."""

    def __eq__(self, other):
        raise AssertionError("__eq__ must not be called when testing for the sentinel")


def test_input_socket_with_an_incomparable_default_is_not_mandatory():
    """`is_mandatory` tests for the sentinel by identity, so it never compares the default it was given."""
    assert not InputSocket(name="value", type=str, default_value=_IncomparableValue()).is_mandatory
