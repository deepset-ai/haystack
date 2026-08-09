# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack.dataclasses import Document
from haystack.utils import FilterBuilder
from haystack.utils.filters import document_matches_filter


def test_single_condition_builds_a_comparison_dict():
    filters = FilterBuilder().eq("meta.type", "article").build()

    assert filters == {"field": "meta.type", "operator": "==", "value": "article"}


def test_multiple_conditions_are_anded():
    filters = FilterBuilder().eq("meta.type", "article").gte("meta.year", 2024).build()

    assert filters == {
        "operator": "AND",
        "conditions": [
            {"field": "meta.type", "operator": "==", "value": "article"},
            {"field": "meta.year", "operator": ">=", "value": 2024},
        ],
    }


def test_or_group_nests_correctly():
    filters = (
        FilterBuilder()
        .eq("meta.type", "article")
        .or_group(lambda f: f.in_("meta.genre", ["economy"]).eq("meta.publisher", "nytimes"))
        .build()
    )

    assert filters == {
        "operator": "AND",
        "conditions": [
            {"field": "meta.type", "operator": "==", "value": "article"},
            {
                "operator": "OR",
                "conditions": [
                    {"field": "meta.genre", "operator": "in", "value": ["economy"]},
                    {"field": "meta.publisher", "operator": "==", "value": "nytimes"},
                ],
            },
        ],
    }


def test_built_filters_are_accepted_by_document_matches_filter():
    """The builder must produce dicts the existing filter engine already understands."""
    document = Document(content="x", meta={"type": "article", "year": 2025})

    filters = FilterBuilder().eq("meta.type", "article").gte("meta.year", 2024).build()

    assert document_matches_filter(filters, document)


def test_empty_builder_raises():
    import pytest

    from haystack.errors import FilterError

    with pytest.raises(FilterError):
        FilterBuilder().build()


def test_builder_operators_stay_in_sync_with_the_filter_engine():
    """The builder must only emit operators `document_matches_filter` understands.

    FilterBuilder spells its operator strings out instead of unpacking
    COMPARISON_OPERATORS, so this is the check that catches a drift between the two.
    """
    from haystack.utils.filter_builder import _EQ, _GT, _GTE, _IN, _LT, _LTE, _NE, _NOT_IN
    from haystack.utils.filters import COMPARISON_OPERATORS

    assert set(COMPARISON_OPERATORS) == {_EQ, _NE, _GT, _GTE, _LT, _LTE, _IN, _NOT_IN}


def test_each_builder_method_emits_the_operator_it_names():
    """Guards against an operator constant being wired to the wrong method."""
    cases = [
        ("eq", "=="),
        ("ne", "!="),
        ("gt", ">"),
        ("gte", ">="),
        ("lt", "<"),
        ("lte", "<="),
        ("in_", "in"),
        ("not_in", "not in"),
    ]
    for method, operator in cases:
        built = getattr(FilterBuilder(), method)("meta.x", 1).build()
        assert built == {"field": "meta.x", "operator": operator, "value": 1}
