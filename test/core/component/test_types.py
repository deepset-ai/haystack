# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import copy
import pickle

import pytest

from haystack.core.component.types import GreedyVariadic, InputSocket, _Empty, _empty
from haystack.core.pipeline.component_checks import (
    _NO_OUTPUT_PRODUCED,
    any_socket_input_received,
    has_socket_received_all_inputs,
)
from haystack.core.pipeline.utils import _deepcopy_with_exceptions
from haystack.utils.base_serialization import _deserialize_value_with_schema, _serialize_value_with_schema


class _AlwaysUnequal:  # noqa: PLW1641  # __hash__ is irrelevant; this only needs a hostile __eq__
    """Stands in for values such as numpy arrays, whose `__eq__` does not return a bool."""

    def __eq__(self, other):
        raise AssertionError("__eq__ must not be called when testing for the sentinel")


class TestEmptySentinel:
    def test_the_sentinel_alias_is_the_singleton(self):
        assert _NO_OUTPUT_PRODUCED is _empty
        assert repr(_empty) == "_empty"

    @pytest.mark.parametrize("copier", [copy.copy, copy.deepcopy, lambda v: pickle.loads(pickle.dumps(v))])
    def test_copying_the_sentinel_returns_the_singleton(self, copier):
        assert copier(_empty) is _empty

    def test_the_sentinel_survives_a_deepcopy_of_the_pipeline_inputs(self):
        """`Pipeline.run` deep-copies its inputs, which carry the sentinel, before snapshotting them."""
        inputs = {"comp": {"socket": [{"sender": "other", "value": _empty}]}}

        copied = _deepcopy_with_exceptions(inputs)

        assert copied["comp"]["socket"][0]["value"] is _empty

    def test_the_sentinel_survives_serialization(self):
        restored = _deserialize_value_with_schema(_serialize_value_with_schema({"value": _empty}))

        assert restored["value"] is _empty


class TestSocketChecksAgainstTheSentinel:
    """The socket checks compare by type, so they hold for any `_Empty` instance and never run `__eq__`."""

    @pytest.mark.parametrize("sentinel", [_empty, _Empty()], ids=["singleton", "separate instance"])
    def test_a_socket_holding_only_the_sentinel_received_nothing(self, sentinel):
        socket = InputSocket(name="value", type=GreedyVariadic[int], senders=["sender"])
        socket_inputs = [{"sender": "sender", "value": sentinel}]

        assert not any_socket_input_received(socket_inputs)
        assert not has_socket_received_all_inputs(socket, socket_inputs)

    def test_a_socket_holding_a_value_received_it(self):
        socket = InputSocket(name="value", type=GreedyVariadic[int], senders=["sender"])
        socket_inputs = [{"sender": "sender", "value": 5}]

        assert any_socket_input_received(socket_inputs)
        assert has_socket_received_all_inputs(socket, socket_inputs)

    def test_a_value_that_cannot_be_compared_counts_as_received(self):
        socket = InputSocket(name="value", type=GreedyVariadic[int], senders=["sender"])
        socket_inputs = [{"sender": "sender", "value": _AlwaysUnequal()}]

        assert any_socket_input_received(socket_inputs)
        assert has_socket_received_all_inputs(socket, socket_inputs)


class TestInputSocketIsMandatory:
    def test_a_socket_without_a_default_is_mandatory(self):
        assert InputSocket(name="value", type=str).is_mandatory

    @pytest.mark.parametrize("default", [None, 0, "", [], False, _AlwaysUnequal()], ids=repr)
    def test_a_socket_with_any_default_is_not_mandatory(self, default):
        assert not InputSocket(name="value", type=str, default_value=default).is_mandatory
