# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from typing import Any, Dict, Optional
from enum import Enum

from haystack.document_stores.types import FilterPolicy
from haystack.document_stores.types.filter_policy import apply_filter_policy


def test_replace_policy_with_both_filters():
    init_filters = {"status": "active", "category": "news"}
    runtime_filters = {"author": "John Doe"}
    result = apply_filter_policy(FilterPolicy.REPLACE, init_filters, runtime_filters)
    assert result == runtime_filters


def test_merge_policy_with_both_filters():
    init_filters = {"status": "active", "category": "news"}
    runtime_filters = {"author": "John Doe"}
    result = apply_filter_policy(FilterPolicy.MERGE, init_filters, runtime_filters)
    assert result == {"status": "active", "category": "news", "author": "John Doe"}


def test_replace_policy_with_only_init_filters():
    init_filters = {"status": "active", "category": "news"}
    runtime_filters = None
    result = apply_filter_policy(FilterPolicy.REPLACE, init_filters, runtime_filters)
    assert result == init_filters


def test_merge_policy_with_only_init_filters():
    init_filters = {"status": "active", "category": "news"}
    runtime_filters = None
    result = apply_filter_policy(FilterPolicy.MERGE, init_filters, runtime_filters)
    assert result == init_filters


def test_merge_policy_with_overlapping_keys():
    init_filters = {"status": "active", "category": "news"}
    runtime_filters = {"category": "science", "author": "John Doe"}
    result = apply_filter_policy(FilterPolicy.MERGE, init_filters, runtime_filters)
    assert result == {"status": "active", "category": "science", "author": "John Doe"}
