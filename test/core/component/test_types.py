# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import copy
import pickle

import pytest

from haystack.core.component.types import InputSocket, _empty
from haystack.core.pipeline.utils import _deepcopy_with_exceptions
from haystack.utils.base_serialization import _deserialize_value_with_schema, _serialize_value_with_schema


class _AlwaysUnequal:  # noqa: PLW1641  # __hash__ is irrelevant; this only needs a hostile __eq__
    """Stands in for values such as numpy arrays, whose `__eq__` does not return a bool."""

    def __eq__(self, other):
        raise AssertionError("__eq__ must not be called when testing for the sentinel")


class TestEmptySentinel:
    """`_empty` is only ever recognized as itself, so anything that could duplicate it has to hand it back."""

    def test_repr_names_the_sentinel(self):
        assert repr(_empty) == "_empty"

    @pytest.mark.parametrize(
        "duplicate",
        [copy.copy, copy.deepcopy, lambda v: pickle.loads(pickle.dumps(v))],
        ids=["copy", "deepcopy", "pickle"],
    )
    def test_duplicating_the_sentinel_returns_the_singleton(self, duplicate):
        assert duplicate(_empty) is _empty

    def test_the_sentinel_survives_a_deepcopy_of_the_pipeline_inputs(self):
        """`Pipeline.run` deep-copies its inputs, which carry the sentinel, before snapshotting them."""
        inputs = {"comp": {"socket": [{"sender": "other", "value": _empty}]}}

        copied = _deepcopy_with_exceptions(inputs)

        assert copied["comp"]["socket"][0]["value"] is _empty

    def test_the_sentinel_survives_serialization(self):
        """A pipeline snapshot serializes the inputs, sentinel included."""
        restored = _deserialize_value_with_schema(_serialize_value_with_schema({"value": _empty}))

        assert restored["value"] is _empty


def test_input_socket_with_an_incomparable_default_is_not_mandatory():
    """`is_mandatory` tests for the sentinel by type, so it never compares the default it was given."""
    assert not InputSocket(name="value", type=str, default_value=_AlwaysUnequal()).is_mandatory
