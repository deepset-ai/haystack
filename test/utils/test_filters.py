import pytest
import pandas as pd

from haystack import Document
from haystack.errors import FilterError
from haystack.utils.filters import convert, document_matches_filter

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
        {"field": "dataframe", "operator": "==", "value": pd.DataFrame([1])},
        Document(dataframe=pd.DataFrame([1])),
        True,
        id="== operator with equal pandas.DataFrame values",
    ),
    pytest.param(
        {"field": "dataframe", "operator": "==", "value": pd.DataFrame([1])},
        Document(dataframe=pd.DataFrame([10])),
        False,
        id="== operator with different pandas.DataFrame values",
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
        {"field": "dataframe", "operator": "!=", "value": pd.DataFrame([1])},
        Document(dataframe=pd.DataFrame([1])),
        False,
        id="!= operator with equal pandas.DataFrame values",
    ),
    pytest.param(
        {"field": "dataframe", "operator": "!=", "value": pd.DataFrame([1])},
        Document(dataframe=pd.DataFrame([10])),
        True,
        id="!= operator with different pandas.DataFrame values",
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
]


@pytest.mark.parametrize("filter, document, expected_result", document_matches_filter_data)
def test_document_matches_filter(filter, document, expected_result):
    assert document_matches_filter(filter, document) == expected_result


document_matches_filter_raises_error_data = [
    # > operator params
    pytest.param({"field": "meta.page", "operator": ">", "value": "10"}, id="> operator with string filter value"),
    pytest.param({"field": "meta.page", "operator": ">", "value": [10]}, id="> operator with list filter value"),
    pytest.param(
        {"field": "meta.page", "operator": ">", "value": pd.DataFrame([10])},
        id="> operator with pandas.DataFrame filter value",
    ),
    # >= operator params
    pytest.param({"field": "meta.page", "operator": ">=", "value": "10"}, id=">= operator with string filter value"),
    pytest.param({"field": "meta.page", "operator": ">=", "value": [10]}, id=">= operator with list filter value"),
    pytest.param(
        {"field": "meta.page", "operator": ">=", "value": pd.DataFrame([10])},
        id=">= operator with pandas.DataFrame filter value",
    ),
    # < operator params
    pytest.param({"field": "meta.page", "operator": "<", "value": "10"}, id="< operator with string filter value"),
    pytest.param({"field": "meta.page", "operator": "<", "value": [10]}, id="< operator with list filter value"),
    pytest.param(
        {"field": "meta.page", "operator": "<", "value": pd.DataFrame([10])},
        id="< operator with pandas.DataFrame filter value",
    ),
    # <= operator params
    pytest.param({"field": "meta.page", "operator": "<=", "value": "10"}, id="<= operator with string filter value"),
    pytest.param({"field": "meta.page", "operator": "<=", "value": [10]}, id="<= operator with list filter value"),
    pytest.param(
        {"field": "meta.page", "operator": "<=", "value": pd.DataFrame([10])},
        id="<= operator with pandas.DataFrame filter value",
    ),
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


@pytest.mark.parametrize("filter", document_matches_filter_raises_error_data)
def test_document_matches_filter_raises_error(filter):
    with pytest.raises(FilterError):
        document = Document(meta={"page": 10})
        document_matches_filter(filter, document)


filters_data = [
    pytest.param(
        {
            "$and": {
                "type": {"$eq": "article"},
                "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
                "rating": {"$gte": 3},
                "$or": {"genre": {"$in": ["economy", "politics"]}, "publisher": {"$eq": "nytimes"}},
            }
        },
        {
            "operator": "AND",
            "conditions": [
                {"field": "type", "operator": "==", "value": "article"},
                {"field": "date", "operator": ">=", "value": "2015-01-01"},
                {"field": "date", "operator": "<", "value": "2021-01-01"},
                {"field": "rating", "operator": ">=", "value": 3},
                {
                    "operator": "OR",
                    "conditions": [
                        {"field": "genre", "operator": "in", "value": ["economy", "politics"]},
                        {"field": "publisher", "operator": "==", "value": "nytimes"},
                    ],
                },
            ],
        },
        id="All operators explicit",
    ),
    pytest.param(
        {
            "type": "article",
            "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
            "rating": {"$gte": 3},
            "$or": {"genre": ["economy", "politics"], "publisher": "nytimes"},
        },
        {
            "operator": "AND",
            "conditions": [
                {"field": "type", "operator": "==", "value": "article"},
                {"field": "date", "operator": ">=", "value": "2015-01-01"},
                {"field": "date", "operator": "<", "value": "2021-01-01"},
                {"field": "rating", "operator": ">=", "value": 3},
                {
                    "operator": "OR",
                    "conditions": [
                        {"field": "genre", "operator": "in", "value": ["economy", "politics"]},
                        {"field": "publisher", "operator": "==", "value": "nytimes"},
                    ],
                },
            ],
        },
        id="Root $and implicit",
    ),
    pytest.param(
        {
            "$or": [
                {"Type": "News Paper", "Date": {"$lt": "2019-01-01"}},
                {"Type": "Blog Post", "Date": {"$gte": "2019-01-01"}},
            ]
        },
        {
            "operator": "OR",
            "conditions": [
                {
                    "operator": "AND",
                    "conditions": [
                        {"field": "Type", "operator": "==", "value": "News Paper"},
                        {"field": "Date", "operator": "<", "value": "2019-01-01"},
                    ],
                },
                {
                    "operator": "AND",
                    "conditions": [
                        {"field": "Type", "operator": "==", "value": "Blog Post"},
                        {"field": "Date", "operator": ">=", "value": "2019-01-01"},
                    ],
                },
            ],
        },
        id="Root $or with list and multiple comparisons",
    ),
    pytest.param(
        {"text": "A Foo Document 1"},
        {"operator": "AND", "conditions": [{"field": "text", "operator": "==", "value": "A Foo Document 1"}]},
        id="Implicit root $and and field $eq",
    ),
    pytest.param(
        {"$or": {"name": {"$or": [{"$eq": "name_0"}, {"$eq": "name_1"}]}, "number": {"$lt": 1.0}}},
        {
            "operator": "OR",
            "conditions": [
                {
                    "operator": "OR",
                    "conditions": [
                        {"field": "name", "operator": "==", "value": "name_0"},
                        {"field": "name", "operator": "==", "value": "name_1"},
                    ],
                },
                {"field": "number", "operator": "<", "value": 1.0},
            ],
        },
        id="Root $or with dict and field $or with list",
    ),
    pytest.param(
        {"number": {"$lte": 2, "$gte": 0}, "name": ["name_0", "name_1"]},
        {
            "operator": "AND",
            "conditions": [
                {"field": "number", "operator": "<=", "value": 2},
                {"field": "number", "operator": ">=", "value": 0},
                {"field": "name", "operator": "in", "value": ["name_0", "name_1"]},
            ],
        },
        id="Implicit $and and field $in",
    ),
    pytest.param(
        {"number": {"$and": [{"$lte": 2}, {"$gte": 0}]}},
        {
            "operator": "AND",
            "conditions": [
                {"field": "number", "operator": "<=", "value": 2},
                {"field": "number", "operator": ">=", "value": 0},
            ],
        },
        id="Implicit root $and and field $and with list",
    ),
    pytest.param(
        {
            "$not": {
                "number": {"$lt": 1.0},
                "$and": {"name": {"$in": ["name_0", "name_1"]}, "$not": {"chapter": {"$eq": "intro"}}},
            }
        },
        {
            "operator": "NOT",
            "conditions": [
                {"field": "number", "operator": "<", "value": 1.0},
                {
                    "operator": "AND",
                    "conditions": [
                        {"field": "name", "operator": "in", "value": ["name_0", "name_1"]},
                        {"operator": "NOT", "conditions": [{"field": "chapter", "operator": "==", "value": "intro"}]},
                    ],
                },
            ],
        },
        id="Root explicit $not",
    ),
    pytest.param(
        {"page": {"$not": 102}},
        {"operator": "NOT", "conditions": [{"field": "page", "operator": "==", "value": 102}]},
        id="Explicit $not with implicit $eq",
    ),
]


@pytest.mark.parametrize("old_style, new_style", filters_data)
def test_convert(old_style, new_style):
    assert convert(old_style) == new_style


def test_convert_with_incorrect_input_type():
    with pytest.raises(ValueError):
        convert("some string")


def test_convert_with_incorrect_filter_nesting():
    with pytest.raises(FilterError):
        convert({"number": {"page": "100"}})

    with pytest.raises(FilterError):
        convert({"number": {"page": {"chapter": "intro"}}})
