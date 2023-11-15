import pytest

from haystack.preview.document_stores.filters import convert


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
]


@pytest.mark.parametrize("old_style, new_style", filters_data)
def test_convert(old_style, new_style):
    assert convert(old_style) == new_style
