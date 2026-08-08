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
