# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack.document_stores.types import apply_filter_policy, FilterPolicy


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
def test_merge_with_custom_logical_operator(logical_operator: str):
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
