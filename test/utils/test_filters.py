# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack import Document
from haystack.errors import FilterError
from haystack.utils.filters import document_matches_filter

document_matches_filter_data = [
    # == operator params
    pytest.param(
        {"field": "meta.name", "operator": "==", "value": "test"},
        Document(meta={"name": "test"}),
        True,
        id="== operator with equal values",
    ),
    pytest.param(
        {"field": "meta.name", "operator": "==", "value": "test"},
        Document(meta={"name": "different value"}),
        False,
        id="== operator with different values",
    ),
    pytest.param(
        {"field": "meta.name", "operator": "==", "value": "test"},
        Document(meta={"name": ["test"]}),
        False,
        id="== operator with different types values",
    ),
    pytest.param(
        {"field": "meta.name", "operator": "==", "value": "test"},
        Document(),
        False,
        id="== operator with missing Document value",
    ),
    pytest.param(
        {"field": "meta.name", "operator": "==", "value": "test"},
        Document(meta={"name": None}),
        False,
        id="== operator with None Document value",
    ),
    pytest.param(
        {"field": "meta.name", "operator": "==", "value": None},
        Document(meta={"name": "test"}),
        False,
        id="== operator with None filter value",
    ),
    # != operator params
    pytest.param(
        {"field": "meta.name", "operator": "!=", "value": "test"},
        Document(meta={"name": "test"}),
        False,
        id="!= operator with equal values",
    ),
    pytest.param(
        {"field": "meta.name", "operator": "!=", "value": "test"},
        Document(meta={"name": "different value"}),
        True,
        id="!= operator with different values",
    ),
    pytest.param(
        {"field": "meta.name", "operator": "!=", "value": "test"},
        Document(meta={"name": ["test"]}),
        True,
        id="!= operator with different types values",
    ),
    pytest.param(
        {"field": "meta.name", "operator": "!=", "value": "test"}, Document(), True, id="!= operator with missing value"
    ),
    pytest.param(
        {"field": "meta.name", "operator": "!=", "value": "test"},
        Document(meta={"name": None}),
        True,
        id="!= operator with None Document value",
    ),
    pytest.param(
        {"field": "meta.name", "operator": "!=", "value": None},
        Document(meta={"name": "test"}),
        True,
        id="!= operator with None filter value",
    ),
    # > operator params
    pytest.param(
        {"field": "meta.page", "operator": ">", "value": 10},
        Document(meta={"page": 10}),
        False,
        id="> operator with equal Document value",
    ),
    pytest.param(
        {"field": "meta.page", "operator": ">", "value": 10},
        Document(meta={"page": 11}),
        True,
        id="> operator with greater Document value",
    ),
    pytest.param(
        {"field": "meta.page", "operator": ">", "value": 10},
        Document(meta={"page": 9}),
        False,
        id="> operator with smaller Document value",
    ),
    pytest.param(
        {"field": "meta.date", "operator": ">", "value": "1969-07-21T20:17:40"},
        Document(meta={"date": "1969-07-21T20:17:40"}),
        False,
        id="> operator with equal ISO 8601 datetime Document value",
    ),
    pytest.param(
        {"field": "meta.date", "operator": ">", "value": "1969-07-21T20:17:40"},
        Document(meta={"date": "1972-12-11T19:54:58"}),
        True,
        id="> operator with greater ISO 8601 datetime Document value",
    ),
    pytest.param(
        {"field": "meta.date", "operator": ">", "value": "1972-12-11T19:54:58"},
        Document(meta={"date": "1969-07-21T20:17:40"}),
        False,
        id="> operator with smaller ISO 8601 datetime Document value",
    ),
    pytest.param(
        {"field": "meta.page", "operator": ">", "value": 10},
        Document(),
        False,
        id="> operator with missing Document value",
    ),
    pytest.param(
        {"field": "meta.page", "operator": ">", "value": 10},
        Document(meta={"page": None}),
        False,
        id="> operator with None Document value",
    ),
    pytest.param(
        {"field": "meta.page", "operator": ">", "value": None},
        Document(meta={"page": 10}),
        False,
        id="> operator with None filter value",
    ),
    pytest.param(
        {"field": "meta.page", "operator": ">", "value": None},
        Document(meta={"page": None}),
        False,
        id="> operator with None Document and filter value",
    ),
    # >= operator params
    pytest.param(
        {"field": "meta.page", "operator": ">=", "value": 10},
        Document(meta={"page": 10}),
        True,
        id=">= operator with equal Document value",
    ),
    pytest.param(
        {"field": "meta.page", "operator": ">=", "value": 10},
        Document(meta={"page": 11}),
        True,
        id=">= operator with greater Document value",
    ),
    pytest.param(
        {"field": "meta.page", "operator": ">=", "value": 10},
        Document(meta={"page": 9}),
        False,
        id=">= operator with smaller Document value",
    ),
    pytest.param(
        {"field": "meta.date", "operator": ">=", "value": "1969-07-21T20:17:40"},
        Document(meta={"date": "1969-07-21T20:17:40"}),
        True,
        id=">= operator with equal ISO 8601 datetime Document value",
    ),
    pytest.param(
        {"field": "meta.date", "operator": ">=", "value": "1969-07-21T20:17:40"},
        Document(meta={"date": "1972-12-11T19:54:58"}),
        True,
        id=">= operator with greater ISO 8601 datetime Document value",
    ),
    pytest.param(
        {"field": "meta.date", "operator": ">=", "value": "1972-12-11T19:54:58"},
        Document(meta={"date": "1969-07-21T20:17:40"}),
        False,
        id=">= operator with smaller ISO 8601 datetime Document value",
    ),
    pytest.param(
        {"field": "meta.page", "operator": ">=", "value": 10},
        Document(),
        False,
        id=">= operator with missing Document value",
    ),
    pytest.param(
        {"field": "meta.page", "operator": ">=", "value": 10},
        Document(meta={"page": None}),
        False,
        id=">= operator with None Document value",
    ),
    pytest.param(
        {"field": "meta.page", "operator": ">=", "value": None},
        Document(meta={"page": 10}),
        False,
        id=">= operator with None filter value",
    ),
    pytest.param(
        {"field": "meta.page", "operator": ">=", "value": None},
        Document(meta={"page": None}),
        False,
        id=">= operator with None Document and filter value",
    ),
    # < operator params
    pytest.param(
        {"field": "meta.page", "operator": "<", "value": 10},
        Document(meta={"page": 10}),
        False,
        id="< operator with equal Document value",
    ),
    pytest.param(
        {"field": "meta.page", "operator": "<", "value": 10},
        Document(meta={"page": 11}),
        False,
        id="< operator with greater Document value",
    ),
    pytest.param(
        {"field": "meta.page", "operator": "<", "value": 10},
        Document(meta={"page": 9}),
        True,
        id="< operator with smaller Document value",
    ),
    pytest.param(
        {"field": "meta.date", "operator": "<", "value": "1969-07-21T20:17:40"},
        Document(meta={"date": "1969-07-21T20:17:40"}),
        False,
        id="< operator with equal ISO 8601 datetime Document value",
    ),
    pytest.param(
        {"field": "meta.date", "operator": "<", "value": "1969-07-21T20:17:40"},
        Document(meta={"date": "1972-12-11T19:54:58"}),
        False,
        id="< operator with greater ISO 8601 datetime Document value",
    ),
    pytest.param(
        {"field": "meta.date", "operator": "<", "value": "1972-12-11T19:54:58"},
        Document(meta={"date": "1969-07-21T20:17:40"}),
        True,
        id="< operator with smaller ISO 8601 datetime Document value",
    ),
    pytest.param(
        {"field": "meta.page", "operator": "<", "value": 10},
        Document(),
        False,
        id="< operator with missing Document value",
    ),
    pytest.param(
        {"field": "meta.page", "operator": "<", "value": 10},
        Document(meta={"page": None}),
        False,
        id="< operator with None Document value",
    ),
    pytest.param(
        {"field": "meta.page", "operator": "<", "value": None},
        Document(meta={"page": 10}),
        False,
        id="< operator with None filter value",
    ),
    pytest.param(
        {"field": "meta.page", "operator": "<", "value": None},
        Document(meta={"page": None}),
        False,
        id="< operator with None Document and filter value",
    ),
    # <= operator params
    pytest.param(
        {"field": "meta.page", "operator": "<=", "value": 10},
        Document(meta={"page": 10}),
        True,
        id="<= operator with equal Document value",
    ),
    pytest.param(
        {"field": "meta.page", "operator": "<=", "value": 10},
        Document(meta={"page": 11}),
        False,
        id="<= operator with greater Document value",
    ),
    pytest.param(
        {"field": "meta.page", "operator": "<=", "value": 10},
        Document(meta={"page": 9}),
        True,
        id="<= operator with smaller Document value",
    ),
    pytest.param(
        {"field": "meta.date", "operator": "<=", "value": "1969-07-21T20:17:40"},
        Document(meta={"date": "1969-07-21T20:17:40"}),
        True,
        id="<= operator with equal ISO 8601 datetime Document value",
    ),
    pytest.param(
        {"field": "meta.date", "operator": "<=", "value": "1969-07-21T20:17:40"},
        Document(meta={"date": "1972-12-11T19:54:58"}),
        False,
        id="<= operator with greater ISO 8601 datetime Document value",
    ),
    pytest.param(
        {"field": "meta.date", "operator": "<=", "value": "1972-12-11T19:54:58"},
        Document(meta={"date": "1969-07-21T20:17:40"}),
        True,
        id="<= operator with smaller ISO 8601 datetime Document value",
    ),
    pytest.param(
        {"field": "meta.page", "operator": "<=", "value": 10},
        Document(),
        False,
        id="<= operator with missing Document value",
    ),
    pytest.param(
        {"field": "meta.page", "operator": "<=", "value": 10},
        Document(meta={"page": None}),
        False,
        id="<= operator with None Document value",
    ),
    pytest.param(
        {"field": "meta.page", "operator": "<=", "value": None},
        Document(meta={"page": 10}),
        False,
        id="<= operator with None filter value",
    ),
    pytest.param(
        {"field": "meta.page", "operator": "<=", "value": None},
        Document(meta={"page": None}),
        False,
        id="<= operator with None Document and filter value",
    ),
    # in operator params
    pytest.param(
        {"field": "meta.page", "operator": "in", "value": [9, 10]},
        Document(meta={"page": 1}),
        False,
        id="in operator with filter value not containing Document value",
    ),
    pytest.param(
        {"field": "meta.page", "operator": "in", "value": [9, 10]},
        Document(meta={"page": 10}),
        True,
        id="in operator with filter value containing Document value",
    ),
    # not in operator params
    pytest.param(
        {"field": "meta.page", "operator": "not in", "value": [9, 10]},
        Document(meta={"page": 1}),
        True,
        id="not in operator with filter value not containing Document value",
    ),
    pytest.param(
        {"field": "meta.page", "operator": "not in", "value": [9, 10]},
        Document(meta={"page": 10}),
        False,
        id="not in operator with filter value containing Document value",
    ),
    # AND operator params
    pytest.param(
        {
            "operator": "AND",
            "conditions": [
                {"field": "meta.page", "operator": "==", "value": 10},
                {"field": "meta.type", "operator": "==", "value": "article"},
            ],
        },
        Document(meta={"page": 10, "type": "article"}),
        True,
        id="AND operator with Document matching all conditions",
    ),
    pytest.param(
        {
            "operator": "AND",
            "conditions": [
                {"field": "meta.page", "operator": "==", "value": 10},
                {"field": "meta.type", "operator": "==", "value": "article"},
            ],
        },
        Document(meta={"page": 20, "type": "article"}),
        False,
        id="AND operator with Document matching a single condition",
    ),
    pytest.param(
        {
            "operator": "AND",
            "conditions": [
                {"field": "meta.page", "operator": "==", "value": 10},
                {"field": "meta.type", "operator": "==", "value": "article"},
            ],
        },
        Document(meta={"page": 11, "value": "blog post"}),
        False,
        id="AND operator with Document matching no condition",
    ),
    # OR operator params
    pytest.param(
        {
            "operator": "OR",
            "conditions": [
                {"field": "meta.page", "operator": "==", "value": 10},
                {"field": "meta.type", "operator": "==", "value": "article"},
            ],
        },
        Document(meta={"page": 10, "type": "article"}),
        True,
        id="OR operator with Document matching all conditions",
    ),
    pytest.param(
        {
            "operator": "OR",
            "conditions": [
                {"field": "meta.page", "operator": "==", "value": 10},
                {"field": "meta.type", "operator": "==", "value": "article"},
            ],
        },
        Document(meta={"page": 20, "type": "article"}),
        True,
        id="OR operator with Document matching a single condition",
    ),
    pytest.param(
        {
            "operator": "OR",
            "conditions": [
                {"field": "meta.page", "operator": "==", "value": 10},
                {"field": "meta.type", "operator": "==", "value": "article"},
            ],
        },
        Document(meta={"page": 11, "value": "blog post"}),
        False,
        id="OR operator with Document matching no condition",
    ),
    # NOT operator params
    pytest.param(
        {
            "operator": "NOT",
            "conditions": [
                {"field": "meta.page", "operator": "==", "value": 10},
                {"field": "meta.type", "operator": "==", "value": "article"},
            ],
        },
        Document(meta={"page": 10, "type": "article"}),
        False,
        id="NOT operator with Document matching all conditions",
    ),
    pytest.param(
        {
            "operator": "NOT",
            "conditions": [
                {"field": "meta.page", "operator": "==", "value": 10},
                {"field": "meta.type", "operator": "==", "value": "article"},
            ],
        },
        Document(meta={"page": 20, "type": "article"}),
        True,
        id="NOT operator with Document matching a single condition",
    ),
    pytest.param(
        {
            "operator": "NOT",
            "conditions": [
                {"field": "meta.page", "operator": "==", "value": 10},
                {"field": "meta.type", "operator": "==", "value": "article"},
            ],
        },
        Document(meta={"page": 11, "value": "blog post"}),
        True,
        id="NOT operator with Document matching no condition",
    ),
    pytest.param(
        {"field": "meta.date", "operator": "==", "value": "2025-02-03T12:45:46.435816Z"},
        Document(meta={"date": "2025-02-03T12:45:46.435816Z"}),
        True,
        id="== operator with ISO 8601 datetime Document value",
    ),
    pytest.param(
        {"field": "meta.date", "operator": ">=", "value": "2025-02-01"},
        Document(meta={"date": "2025-02-03T12:45:46.435816Z"}),
        True,
        id=">= operator with naive and aware ISO 8601 datetime Document value",
    ),
]


@pytest.mark.parametrize("filters, document, expected_result", document_matches_filter_data)
def test_document_matches_filter(filters, document, expected_result):
    assert document_matches_filter(filters, document) == expected_result


document_matches_filter_raises_error_data = [
    # > operator params
    pytest.param({"field": "meta.page", "operator": ">", "value": "10"}, id="> operator with string filter value"),
    pytest.param({"field": "meta.page", "operator": ">", "value": [10]}, id="> operator with list filter value"),
    # >= operator params
    pytest.param({"field": "meta.page", "operator": ">=", "value": "10"}, id=">= operator with string filter value"),
    pytest.param({"field": "meta.page", "operator": ">=", "value": [10]}, id=">= operator with list filter value"),
    # < operator params
    pytest.param({"field": "meta.page", "operator": "<", "value": "10"}, id="< operator with string filter value"),
    pytest.param({"field": "meta.page", "operator": "<", "value": [10]}, id="< operator with list filter value"),
    # <= operator params
    pytest.param({"field": "meta.page", "operator": "<=", "value": "10"}, id="<= operator with string filter value"),
    pytest.param({"field": "meta.page", "operator": "<=", "value": [10]}, id="<= operator with list filter value"),
    # in operator params
    pytest.param({"field": "meta.page", "operator": "in", "value": 1}, id="in operator with non list filter value"),
    # at some point we might want to support any iterable and this test should fail
    pytest.param(
        {"field": "meta.page", "operator": "in", "value": (10, 11)}, id="in operator with non list filter value"
    ),
    # not in operator params
    pytest.param(
        {"field": "meta.page", "operator": "not in", "value": 1}, id="not in operator with non list filter value"
    ),
    # at some point we might want to support any iterable and this test should fail
    pytest.param(
        {"field": "meta.page", "operator": "not in", "value": (10, 11)}, id="not in operator with non list filter value"
    ),
    # Malformed filters
    pytest.param(
        {"conditions": [{"field": "meta.name", "operator": "==", "value": "test"}]}, id="Missing root operator key"
    ),
    pytest.param({"operator": "AND"}, id="Missing root conditions key"),
    pytest.param({"operator": "==", "value": "test"}, id="Missing condition field key"),
    pytest.param({"field": "meta.name", "value": "test"}, id="Missing condition operator key"),
    pytest.param({"field": "meta.name", "operator": "=="}, id="Missing condition value key"),
]


@pytest.mark.parametrize("filters", document_matches_filter_raises_error_data)
def test_document_matches_filter_raises_error(filters):
    with pytest.raises(FilterError):
        document = Document(meta={"page": 10})
        document_matches_filter(filters, document)
