import pytest
import pandas as pd
import numpy as np

from haystack.preview import Document
from haystack.preview.errors import FilterError
from haystack.preview.utils.filters import convert, document_matches_filter


class TestFilterUtils:  # pylint: disable=R0904
    @pytest.mark.unit
    def test_eq_match(self):
        document = Document(meta={"name": "test"})
        filter = {"name": "test"}
        assert document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_eq_no_match(self):
        document = Document(meta={"name": "test"})
        filter = {"name": "test1"}
        assert not document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_eq_no_match_missing_key(self):
        document = Document(meta={"name": "test"})
        filter = {"name1": "test"}
        assert not document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_explicit_eq(self):
        document = Document(meta={"name": "test"})
        filter = {"name": {"$eq": "test"}}
        assert document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_eq_different_types(self):
        document = Document(meta={"name": 1})
        filter = {"name": "1"}
        assert not document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_eq_dataframes(self):
        document = Document(meta={"name": pd.DataFrame({"a": [1, 2, 3]})})
        filter = {"name": pd.DataFrame({"a": [1, 2, 3]})}
        assert document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_eq_dataframes_no_match(self):
        document = Document(meta={"name": pd.DataFrame({"a": [1, 2, 3]})})
        filter = {"name": pd.DataFrame({"a": [1, 2, 4]})}
        assert not document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_eq_np_arrays(self):
        document = Document(meta={"name": np.array([1, 2, 3])})
        filter = {"name": np.array([1, 2, 3])}
        assert document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_eq_np_arrays_no_match(self):
        document = Document(meta={"name": np.array([1, 2, 3])})
        filter = {"name": np.array([1, 2, 4])}
        assert not document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_ne_match(self):
        document = Document(meta={"name": "test"})
        filter = {"name": {"$ne": "test1"}}
        assert document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_ne_no_match(self):
        document = Document(meta={"name": "test"})
        filter = {"name": {"$ne": "test"}}
        assert not document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_ne_no_match_missing_key(self):
        document = Document(meta={"name": "test"})
        filter = {"name1": {"$ne": "test"}}
        assert document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_ne_different_types(self):
        document = Document(meta={"name": 1})
        filter = {"name": {"$ne": "1"}}
        assert document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_ne_dataframes(self):
        document = Document(meta={"name": pd.DataFrame({"a": [1, 2, 3]})})
        filter = {"name": {"$ne": pd.DataFrame({"a": [1, 2, 4]})}}
        assert document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_ne_dataframes_no_match(self):
        document = Document(meta={"name": pd.DataFrame({"a": [1, 2, 3]})})
        filter = {"name": {"$ne": pd.DataFrame({"a": [1, 2, 3]})}}
        assert not document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_ne_np_arrays(self):
        document = Document(meta={"name": np.array([1, 2, 3])})
        filter = {"name": {"$ne": np.array([1, 2, 4])}}
        assert document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_ne_np_arrays_no_match(self):
        document = Document(meta={"name": np.array([1, 2, 3])})
        filter = {"name": {"$ne": np.array([1, 2, 3])}}
        assert not document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_in_match_list(self):
        document = Document(meta={"name": "test"})
        filter = {"name": {"$in": ["test", "test1"]}}
        assert document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_in_no_match_list(self):
        document = Document(meta={"name": "test"})
        filter = {"name": {"$in": ["test2", "test3"]}}
        assert not document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_in_implicit(self):
        document = Document(meta={"name": "test"})
        filter = {"name": ["test", "test1"]}
        assert document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_in_match_set(self):
        document = Document(meta={"name": "test"})
        filter = {"name": {"$in": {"test", "test1"}}}
        assert document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_in_no_match_set(self):
        document = Document(meta={"name": "test"})
        filter = {"name": {"$in": {"test2", "test3"}}}
        assert not document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_in_match_tuple(self):
        document = Document(meta={"name": "test"})
        filter = {"name": {"$in": ("test", "test1")}}
        assert document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_in_no_match_tuple(self):
        document = Document(meta={"name": "test"})
        filter = {"name": {"$in": ("test2", "test3")}}
        assert not document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_in_no_match_missing_key(self):
        document = Document(meta={"name": "test"})
        filter = {"name1": {"$in": ["test", "test1"]}}
        assert not document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_in_unsupported_type(self):
        document = Document(meta={"name": "test"})
        filter = {"name": {"$in": "unsupported"}}
        with pytest.raises(FilterError, match=r"\$in accepts only iterable values like lists, sets and tuples"):
            document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_nin_match_list(self):
        document = Document(meta={"name": "test"})
        filter = {"name": {"$nin": ["test1", "test2"]}}
        assert document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_nin_no_match_list(self):
        document = Document(meta={"name": "test"})
        filter = {"name": {"$nin": ["test", "test1"]}}
        assert not document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_nin_match_set(self):
        document = Document(meta={"name": "test"})
        filter = {"name": {"$nin": {"test1", "test2"}}}
        assert document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_nin_no_match_set(self):
        document = Document(meta={"name": "test"})
        filter = {"name": {"$nin": {"test", "test1"}}}
        assert not document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_nin_match_tuple(self):
        document = Document(meta={"name": "test"})
        filter = {"name": {"$nin": ("test1", "test2")}}
        assert document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_nin_no_match_tuple(self):
        document = Document(meta={"name": "test"})
        filter = {"name": {"$nin": ("test", "test1")}}
        assert not document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_nin_no_match_missing_key(self):
        document = Document(meta={"name": "test"})
        filter = {"name1": {"$nin": ["test", "test1"]}}
        assert document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_nin_unsupported_type(self):
        document = Document(meta={"name": "test"})
        filter = {"name": {"$nin": "unsupported"}}
        with pytest.raises(FilterError, match=r"\$in accepts only iterable values like lists, sets and tuples"):
            document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_gt_match_int(self):
        document = Document(meta={"age": 21})
        filter = {"age": {"$gt": 20}}
        assert document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_gt_no_match_int(self):
        document = Document(meta={"age": 19})
        filter = {"age": {"$gt": 20}}
        assert not document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_gt_match_float(self):
        document = Document(meta={"number": 90.5})
        filter = {"number": {"$gt": 90.0}}
        assert document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_gt_no_match_float(self):
        document = Document(meta={"number": 89.5})
        filter = {"number": {"$gt": 90.0}}
        assert not document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_gt_match_np_number(self):
        document = Document(meta={"value": np.float64(7.5)})
        filter = {"value": {"$gt": np.float64(7.0)}}
        assert document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_gt_no_match_np_number(self):
        document = Document(meta={"value": np.float64(6.5)})
        filter = {"value": {"$gt": np.float64(7.0)}}
        assert not document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_gt_match_date_string(self):
        document = Document(meta={"date": "2022-01-02"})
        filter = {"date": {"$gt": "2022-01-01"}}
        assert document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_gt_no_match_date_string(self):
        document = Document(meta={"date": "2022-01-01"})
        filter = {"date": {"$gt": "2022-01-01"}}
        assert not document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_gt_no_match_missing_key(self):
        document = Document(meta={"age": 21})
        filter = {"age1": {"$gt": 20}}
        assert not document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_gt_unsupported_type(self):
        document = Document(meta={"age": 21})
        filter = {"age": {"$gt": "unsupported"}}
        with pytest.raises(
            FilterError,
            match=(
                r"Convert these values into one of the following types: \['int', 'float', 'number'\] or a datetime string "
                "in ISO 8601 format"
            ),
        ):
            document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_gte_match_int(self):
        document = Document(meta={"age": 21})
        filter_1 = {"age": {"$gte": 21}}
        filter_2 = {"age": {"$gte": 20}}
        assert document_matches_filter(filter_1, document)
        assert document_matches_filter(filter_2, document)

    @pytest.mark.unit
    def test_gte_no_match_int(self):
        document = Document(meta={"age": 20})
        filter = {"age": {"$gte": 21}}
        assert not document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_gte_match_float(self):
        document = Document(meta={"number": 90.5})
        filter_1 = {"number": {"$gte": 90.5}}
        filter_2 = {"number": {"$gte": 90.4}}
        assert document_matches_filter(filter_1, document)
        assert document_matches_filter(filter_2, document)

    @pytest.mark.unit
    def test_gte_no_match_float(self):
        document = Document(meta={"number": 90.4})
        filter = {"number": {"$gte": 90.5}}
        assert not document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_gte_match_np_number(self):
        document = Document(meta={"value": np.float64(7.5)})
        filter_1 = {"value": {"$gte": np.float64(7.5)}}
        filter_2 = {"value": {"$gte": np.float64(7.4)}}
        assert document_matches_filter(filter_1, document)
        assert document_matches_filter(filter_2, document)

    @pytest.mark.unit
    def test_gte_no_match_np_number(self):
        document = Document(meta={"value": np.float64(7.4)})
        filter = {"value": {"$gte": np.float64(7.5)}}
        assert not document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_gte_match_date_string(self):
        document = Document(meta={"date": "2022-01-02"})
        filter_1 = {"date": {"$gte": "2022-01-02"}}
        filter_2 = {"date": {"$gte": "2022-01-01"}}
        assert document_matches_filter(filter_1, document)
        assert document_matches_filter(filter_2, document)

    @pytest.mark.unit
    def test_gte_no_match_date_string(self):
        document = Document(meta={"date": "2022-01-01"})
        filter = {"date": {"$gte": "2022-01-02"}}
        assert not document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_gte_unsupported_type(self):
        document = Document(meta={"age": 21})
        filter = {"age": {"$gte": "unsupported"}}
        with pytest.raises(
            FilterError,
            match=(
                r"Convert these values into one of the following types: \['int', 'float', 'number'\] or a datetime string "
                "in ISO 8601 format"
            ),
        ):
            document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_lt_match_int(self):
        document = Document(meta={"age": 19})
        filter = {"age": {"$lt": 20}}
        assert document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_lt_no_match_int(self):
        document = Document(meta={"age": 20})
        filter = {"age": {"$lt": 20}}
        assert not document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_lt_match_float(self):
        document = Document(meta={"number": 89.9})
        filter = {"number": {"$lt": 90.0}}
        assert document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_lt_no_match_float(self):
        document = Document(meta={"number": 90.0})
        filter = {"number": {"$lt": 90.0}}
        assert not document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_lt_match_np_number(self):
        document = Document(meta={"value": np.float64(6.9)})
        filter = {"value": {"$lt": np.float64(7.0)}}
        assert document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_lt_no_match_np_number(self):
        document = Document(meta={"value": np.float64(7.0)})
        filter = {"value": {"$lt": np.float64(7.0)}}
        assert not document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_lt_match_date_string(self):
        document = Document(meta={"date": "2022-01-01"})
        filter = {"date": {"$lt": "2022-01-02"}}
        assert document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_lt_no_match_date_string(self):
        document = Document(meta={"date": "2022-01-02"})
        filter = {"date": {"$lt": "2022-01-02"}}
        assert not document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_lt_unsupported_type(self):
        document = Document(meta={"age": 21})
        filter = {"age": {"$lt": "unsupported"}}
        with pytest.raises(
            FilterError,
            match=(
                r"Convert these values into one of the following types: \['int', 'float', 'number'\] or a datetime string "
                "in ISO 8601 format"
            ),
        ):
            document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_lte_match_int(self):
        document = Document(meta={"age": 21})
        filter_1 = {"age": {"$lte": 21}}
        filter_2 = {"age": {"$lte": 20}}
        assert not document_matches_filter(filter_2, document)
        assert document_matches_filter(filter_1, document)

    @pytest.mark.unit
    def test_lte_no_match_int(self):
        document = Document(meta={"age": 22})
        filter = {"age": {"$lte": 21}}
        assert not document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_lte_match_float(self):
        document = Document(meta={"number": 90.5})
        filter_1 = {"number": {"$lte": 90.5}}
        filter_2 = {"number": {"$lte": 90.4}}
        assert not document_matches_filter(filter_2, document)
        assert document_matches_filter(filter_1, document)

    @pytest.mark.unit
    def test_lte_no_match_float(self):
        document = Document(meta={"number": 90.6})
        filter = {"number": {"$lte": 90.5}}
        assert not document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_lte_match_np_number(self):
        document = Document(meta={"value": np.float64(7.5)})
        filter_1 = {"value": {"$lte": np.float64(7.5)}}
        filter_2 = {"value": {"$lte": np.float64(7.4)}}
        assert not document_matches_filter(filter_2, document)
        assert document_matches_filter(filter_1, document)

    @pytest.mark.unit
    def test_lte_no_match_np_number(self):
        document = Document(meta={"value": np.float64(7.6)})
        filter = {"value": {"$lte": np.float64(7.5)}}
        assert not document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_lte_match_date_string(self):
        document = Document(meta={"date": "2022-01-02"})
        filter_1 = {"date": {"$lte": "2022-01-02"}}
        filter_2 = {"date": {"$lte": "2022-01-01"}}
        assert not document_matches_filter(filter_2, document)
        assert document_matches_filter(filter_1, document)

    @pytest.mark.unit
    def test_lte_no_match_date_string(self):
        document = Document(meta={"date": "2022-01-03"})
        filter = {"date": {"$lte": "2022-01-02"}}
        assert not document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_lte_unsupported_type(self):
        document = Document(meta={"age": 21})
        filter = {"age": {"$lte": "unsupported"}}
        with pytest.raises(
            FilterError,
            match=(
                r"Convert these values into one of the following types: \['int', 'float', 'number'\] or a datetime string "
                "in ISO 8601 format"
            ),
        ):
            document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_implicit_and(self):
        document = Document(meta={"age": 21, "name": "John"})
        filter = {"age": {"$gt": 18}, "name": "John"}
        assert document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_explicit_and(self):
        document = Document(meta={"age": 21})
        filter = {"age": {"$and": {"$gt": 18}, "$lt": 25}}
        assert document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_or(self):
        document = Document(meta={"age": 26})
        filter = {"age": {"$or": [{"$gt": 18}, {"$lt": 25}]}}
        assert document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_not(self):
        document = Document(meta={"age": 17})
        filter = {"age": {"$not": {"$gt": 18}}}
        assert document_matches_filter(filter, document)


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
