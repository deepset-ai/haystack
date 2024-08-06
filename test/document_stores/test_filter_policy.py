# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from typing import Any, Dict, Optional
from enum import Enum

from haystack.document_stores.types import FilterPolicy
from haystack.document_stores.types.filter_policy import apply_filter_policy


def test_replace_policy_with_both_filters():
    """
    Replacing legacy filters
    Result: The runtime filters
    """
    init_filters = {"status": "active", "category": "news"}
    runtime_filters = {"author": "John Doe"}
    result = apply_filter_policy(FilterPolicy.REPLACE, init_filters, runtime_filters)
    assert result == runtime_filters


def test_merge_policy_with_both_filters():
    """
    Merging legacy filters
    Result: The runtime filters
    """
    init_filters = {"status": "active", "category": "news"}
    runtime_filters = {"author": "John Doe"}
    result = apply_filter_policy(FilterPolicy.MERGE, init_filters, runtime_filters)
    assert result == {"status": "active", "category": "news", "author": "John Doe"}


def test_replace_policy_with_only_init_filters():
    """
    Replacing legacy filters, None runtime filters
    Result: The init filters
    """
    init_filters = {"status": "active", "category": "news"}
    runtime_filters = None
    result = apply_filter_policy(FilterPolicy.REPLACE, init_filters, runtime_filters)
    assert result == init_filters


def test_merge_policy_with_only_init_filters():
    """
    Merging of legacy filters, None runtime filters
    Result: The init filters
    """
    init_filters = {"status": "active", "category": "news"}
    runtime_filters = None
    result = apply_filter_policy(FilterPolicy.MERGE, init_filters, runtime_filters)
    assert result == init_filters


def test_merge_policy_with_overlapping_keys():
    """
    Merging of legacy filters
    Result: The runtime filters overwrite the init filters
    """
    init_filters = {"status": "active", "category": "news"}
    runtime_filters = {"category": "science", "author": "John Doe"}
    result = apply_filter_policy(FilterPolicy.MERGE, init_filters, runtime_filters)
    assert result == {"status": "active", "category": "science", "author": "John Doe"}


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
def test_merge_with_custom_logical_operator(logical_operator):
    """
    Merging with a custom logical operator
    Result: The given logical operator with both filters
    """
    init_filters = {"field": "meta.date", "operator": ">=", "value": "2015-01-01"}
    runtime_filters = {"field": "meta.type", "operator": "==", "value": "article"}
    result = apply_filter_policy(FilterPolicy.MERGE, init_filters, runtime_filters, logical_operator=logical_operator)
    assert result == {
        "operator": logical_operator,
        "conditions": [
            {"field": "meta.date", "operator": ">=", "value": "2015-01-01"},
            {"field": "meta.type", "operator": "==", "value": "article"},
        ],
    }
