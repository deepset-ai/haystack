# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

import pytest

from haystack import Document
from haystack.document_stores.types import FilterPolicy, apply_filter_policy
from haystack.utils.filters import document_matches_filter


def test_merge_two_comparison_filters():
    """
    Merging two comparison filters

    Result: AND operator with both filters
    """
    init_filters = {"field": "meta.date", "operator": ">=", "value": "2015-01-01"}
    runtime_filters = {"field": "meta.type", "operator": "==", "value": "article"}
    result = apply_filter_policy(FilterPolicy.MERGE, init_filters, runtime_filters)
    assert result == {
        "operator": "AND",
        "conditions": [
            {"field": "meta.date", "operator": ">=", "value": "2015-01-01"},
            {"field": "meta.type", "operator": "==", "value": "article"},
        ],
    }


def test_merge_init_comparison_and_runtime_logical_filters():
    """
    Merging init comparison and runtime logical filters

    Result: AND operator with both filters
    """
    init_filters = {"field": "meta.date", "operator": ">=", "value": "2015-01-01"}
    runtime_filters = {
        "operator": "AND",
        "conditions": [
            {"field": "meta.type", "operator": "==", "value": "article"},
            {"field": "meta.rating", "operator": ">=", "value": 3},
        ],
    }
    result = apply_filter_policy(FilterPolicy.MERGE, init_filters, runtime_filters)
    assert result == {
        "operator": "AND",
        "conditions": [
            {"field": "meta.type", "operator": "==", "value": "article"},
            {"field": "meta.rating", "operator": ">=", "value": 3},
            {"field": "meta.date", "operator": ">=", "value": "2015-01-01"},
        ],
    }


def test_merge_runtime_comparison_and_init_logical_filters_with_string_operators():
    """
    Merging a runtime comparison filter with an init logical filter, but with string-based logical operators

    Result: AND operator with both filters
    """
    # Test with string-based logical operators
    init_filters = {
        "operator": "AND",
        "conditions": [
            {"field": "meta.type", "operator": "==", "value": "article"},
            {"field": "meta.rating", "operator": ">=", "value": 3},
        ],
    }
    runtime_filters = {
        "operator": "AND",
        "conditions": [
            {"field": "meta.genre", "operator": "IN", "value": ["economy", "politics"]},
            {"field": "meta.publisher", "operator": "==", "value": "nytimes"},
        ],
    }
    result = apply_filter_policy(FilterPolicy.MERGE, init_filters, runtime_filters)
    assert result == {
        "operator": "AND",
        "conditions": [
            {"field": "meta.type", "operator": "==", "value": "article"},
            {"field": "meta.rating", "operator": ">=", "value": 3},
            {"field": "meta.genre", "operator": "IN", "value": ["economy", "politics"]},
            {"field": "meta.publisher", "operator": "==", "value": "nytimes"},
        ],
    }


def test_merge_runtime_comparison_and_init_logical_filters():
    """
    Merging a runtime comparison filter with an init logical filter

    Result: AND operator with both filters
    """
    init_filters = {
        "operator": "AND",
        "conditions": [
            {"field": "meta.type", "operator": "==", "value": "article"},
            {"field": "meta.rating", "operator": ">=", "value": 3},
        ],
    }
    runtime_filters = {"field": "meta.date", "operator": ">=", "value": "2015-01-01"}
    result = apply_filter_policy(FilterPolicy.MERGE, init_filters, runtime_filters)
    assert result == {
        "operator": "AND",
        "conditions": [
            {"field": "meta.type", "operator": "==", "value": "article"},
            {"field": "meta.rating", "operator": ">=", "value": 3},
            {"field": "meta.date", "operator": ">=", "value": "2015-01-01"},
        ],
    }


def test_merge_two_logical_filters():
    """
    Merging two logical filters

    Result: AND operator with both filters
    """
    init_filters = {
        "operator": "AND",
        "conditions": [
            {"field": "meta.type", "operator": "==", "value": "article"},
            {"field": "meta.rating", "operator": ">=", "value": 3},
        ],
    }
    runtime_filters = {
        "operator": "AND",
        "conditions": [
            {"field": "meta.genre", "operator": "IN", "value": ["economy", "politics"]},
            {"field": "meta.publisher", "operator": "==", "value": "nytimes"},
        ],
    }
    result = apply_filter_policy(FilterPolicy.MERGE, init_filters, runtime_filters)
    assert result == {
        "operator": "AND",
        "conditions": [
            {"field": "meta.type", "operator": "==", "value": "article"},
            {"field": "meta.rating", "operator": ">=", "value": 3},
            {"field": "meta.genre", "operator": "IN", "value": ["economy", "politics"]},
            {"field": "meta.publisher", "operator": "==", "value": "nytimes"},
        ],
    }


def test_merge_does_not_mutate_logical_filters():
    init_filters = {"operator": "AND", "conditions": [{"field": "meta.type", "operator": "==", "value": "article"}]}
    runtime_filters = {"field": "meta.year", "operator": "==", "value": 2020}

    result = apply_filter_policy(FilterPolicy.MERGE, init_filters, runtime_filters)

    assert result == {
        "operator": "AND",
        "conditions": [
            {"field": "meta.type", "operator": "==", "value": "article"},
            {"field": "meta.year", "operator": "==", "value": 2020},
        ],
    }
    assert init_filters == {
        "operator": "AND",
        "conditions": [{"field": "meta.type", "operator": "==", "value": "article"}],
    }
    assert runtime_filters == {"field": "meta.year", "operator": "==", "value": 2020}


def test_merge_does_not_mutate_runtime_logical_filter():
    init_filters = {"field": "meta.type", "operator": "==", "value": "article"}
    runtime_filters = {"operator": "AND", "conditions": [{"field": "meta.year", "operator": "==", "value": 2020}]}

    result = apply_filter_policy(FilterPolicy.MERGE, init_filters, runtime_filters)

    assert result == {
        "operator": "AND",
        "conditions": [
            {"field": "meta.year", "operator": "==", "value": 2020},
            {"field": "meta.type", "operator": "==", "value": "article"},
        ],
    }
    assert runtime_filters == {
        "operator": "AND",
        "conditions": [{"field": "meta.year", "operator": "==", "value": 2020}],
    }


def test_merge_with_different_logical_operators():
    """
    Merging with a different logical operator

    Result: warnings and runtime filters
    """
    init_filters = {"operator": "AND", "conditions": [{"field": "meta.type", "operator": "==", "value": "article"}]}
    runtime_filters = {
        "operator": "OR",
        "conditions": [{"field": "meta.genre", "operator": "IN", "value": ["economy", "politics"]}],
    }
    result = apply_filter_policy(FilterPolicy.MERGE, init_filters, runtime_filters)
    assert result == runtime_filters


def test_merge_comparison_filters_with_same_field():
    """
    Merging comparison filters with the same field

    Result: warnings and runtime filters
    """
    init_filters = {"field": "meta.date", "operator": ">=", "value": "2015-01-01"}
    runtime_filters = {"field": "meta.date", "operator": "<=", "value": "2020-12-31"}
    result = apply_filter_policy(FilterPolicy.MERGE, init_filters, runtime_filters)
    assert result == runtime_filters


@pytest.mark.parametrize("logical_operator", ["AND", "OR", "NOT"])
def test_merge_with_custom_logical_operator(logical_operator: Literal["AND", "OR", "NOT"]) -> None:
    """
    Merging with a custom logical operator

    Result: The given logical operator with both filters
    """
    init_filters = {"field": "meta.date", "operator": ">=", "value": "2015-01-01"}
    runtime_filters = {"field": "meta.type", "operator": "==", "value": "article"}
    result = apply_filter_policy(
        FilterPolicy.MERGE, init_filters, runtime_filters, default_logical_operator=logical_operator
    )
    assert result == {
        "operator": logical_operator,
        "conditions": [
            {"field": "meta.date", "operator": ">=", "value": "2015-01-01"},
            {"field": "meta.type", "operator": "==", "value": "article"},
        ],
    }


@pytest.mark.parametrize("operator", ["OR", "NOT"])
def test_merge_two_logical_filters_with_non_and_operator_keeps_both_restrictions(
    operator: Literal["OR", "NOT"],
) -> None:
    """
    Merging two logical filters that share a non-AND operator

    Result: both filters nested under AND, so each restriction still applies.
    Concatenating their conditions instead would union them (OR) or negate their
    conjunction (NOT), potentially producing a broader result than applying both filters.
    """
    init_filters = {"operator": operator, "conditions": [{"field": "meta.type", "operator": "==", "value": "article"}]}
    runtime_filters = {
        "operator": operator,
        "conditions": [{"field": "meta.genre", "operator": "==", "value": "economy"}],
    }
    result = apply_filter_policy(FilterPolicy.MERGE, init_filters, runtime_filters)
    assert result == {"operator": "AND", "conditions": [init_filters, runtime_filters]}


@pytest.mark.parametrize("operator", ["OR", "NOT"])
def test_merge_two_logical_filters_with_non_and_operator_matches_the_intersection(
    operator: Literal["OR", "NOT"],
) -> None:
    """
    The merged filter selects exactly the documents both input filters select
    """
    documents = [
        Document(id=str(index), meta=meta)
        for index, meta in enumerate([{"a": 1, "b": 1}, {"a": 1, "b": 9}, {"a": 9, "b": 1}, {"a": 9, "b": 9}])
    ]
    init_filters = {
        "operator": operator,
        "conditions": [
            {"field": "meta.a", "operator": "==", "value": 1},
            {"field": "meta.b", "operator": "==", "value": 1},
        ],
    }
    runtime_filters = {
        "operator": operator,
        "conditions": [
            {"field": "meta.a", "operator": "==", "value": 9},
            {"field": "meta.b", "operator": "==", "value": 9},
        ],
    }
    merged = apply_filter_policy(FilterPolicy.MERGE, init_filters, runtime_filters)
    assert merged is not None

    selected = [doc.id for doc in documents if document_matches_filter(merged, doc)]
    intersection = [
        doc.id
        for doc in documents
        if document_matches_filter(init_filters, doc) and document_matches_filter(runtime_filters, doc)
    ]
    assert selected == intersection


def test_merge_two_and_logical_filters_still_flattens_conditions() -> None:
    """
    Merging two AND logical filters

    Result: a single AND with the conditions of both, unchanged from before
    """
    init_filters = {"operator": "AND", "conditions": [{"field": "meta.type", "operator": "==", "value": "article"}]}
    runtime_filters = {"operator": "AND", "conditions": [{"field": "meta.genre", "operator": "==", "value": "economy"}]}
    result = apply_filter_policy(FilterPolicy.MERGE, init_filters, runtime_filters)
    assert result == {
        "operator": "AND",
        "conditions": [
            {"field": "meta.type", "operator": "==", "value": "article"},
            {"field": "meta.genre", "operator": "==", "value": "economy"},
        ],
    }
