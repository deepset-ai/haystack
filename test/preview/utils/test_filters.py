import pytest
import pandas as pd
import numpy as np

from haystack.preview import Document
from haystack.preview.errors import FilterError
from haystack.preview.utils.filters import document_matches_filter, _find_nested_value


class TestFilterUtils:
    @pytest.mark.unit
    def test_find_nested_value(self):
        assert _find_nested_value({"a": {"b": 1}}, "a.b") == 1
        with pytest.raises(ValueError):
            assert _find_nested_value({"a": {"b": 1}}, "a.c") == None
        with pytest.raises(ValueError):
            assert _find_nested_value({"a": [1]}, "a.b") == None

    @pytest.mark.unit
    def test_eq_match(self):
        document = Document(metadata={"name": "test"})
        filter = {"metadata.name": "test"}
        assert document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_eq_no_match(self):
        document = Document(metadata={"name": "test"})
        filter = {"metadata.name": "test1"}
        assert not document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_eq_no_match_missing_key(self):
        document = Document(metadata={"name": "test"})
        filter = {"metadata.name1": "test"}
        assert not document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_explicit_eq(self):
        document = Document(metadata={"name": "test"})
        filter = {"metadata.name": {"$eq": "test"}}
        assert document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_eq_different_types(self):
        document = Document(metadata={"name": 1})
        filter = {"metadata.name": "1"}
        assert not document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_eq_dataframes(self):
        document = Document(metadata={"name": pd.DataFrame({"a": [1, 2, 3]})})
        filter = {"metadata.name": pd.DataFrame({"a": [1, 2, 3]})}
        assert document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_eq_dataframes_no_match(self):
        document = Document(metadata={"name": pd.DataFrame({"a": [1, 2, 3]})})
        filter = {"metadata.name": pd.DataFrame({"a": [1, 2, 4]})}
        assert not document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_eq_np_arrays(self):
        document = Document(metadata={"name": np.array([1, 2, 3])})
        filter = {"metadata.name": np.array([1, 2, 3])}
        assert document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_eq_np_arrays_no_match(self):
        document = Document(metadata={"name": np.array([1, 2, 3])})
        filter = {"metadata.name": np.array([1, 2, 4])}
        assert not document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_ne_match(self):
        document = Document(metadata={"name": "test"})
        filter = {"metadata.name": {"$ne": "test1"}}
        assert document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_ne_no_match(self):
        document = Document(metadata={"name": "test"})
        filter = {"metadata.name": {"$ne": "test"}}
        assert not document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_ne_no_match_missing_key(self):
        document = Document(metadata={"name": "test"})
        filter = {"metadata.name1": {"$ne": "test"}}
        assert document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_ne_different_types(self):
        document = Document(metadata={"name": 1})
        filter = {"metadata.name": {"$ne": "1"}}
        assert document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_ne_dataframes(self):
        document = Document(metadata={"name": pd.DataFrame({"a": [1, 2, 3]})})
        filter = {"metadata.name": {"$ne": pd.DataFrame({"a": [1, 2, 4]})}}
        assert document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_ne_dataframes_no_match(self):
        document = Document(metadata={"name": pd.DataFrame({"a": [1, 2, 3]})})
        filter = {"metadata.name": {"$ne": pd.DataFrame({"a": [1, 2, 3]})}}
        assert not document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_ne_np_arrays(self):
        document = Document(metadata={"name": np.array([1, 2, 3])})
        filter = {"metadata.name": {"$ne": np.array([1, 2, 4])}}
        assert document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_ne_np_arrays_no_match(self):
        document = Document(metadata={"name": np.array([1, 2, 3])})
        filter = {"metadata.name": {"$ne": np.array([1, 2, 3])}}
        assert not document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_in_match_list(self):
        document = Document(metadata={"name": "test"})
        filter = {"metadata.name": {"$in": ["test", "test1"]}}
        assert document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_in_no_match_list(self):
        document = Document(metadata={"name": "test"})
        filter = {"metadata.name": {"$in": ["test2", "test3"]}}
        assert not document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_in_implicit(self):
        document = Document(metadata={"name": "test"})
        filter = {"metadata.name": ["test", "test1"]}
        assert document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_in_match_set(self):
        document = Document(metadata={"name": "test"})
        filter = {"metadata.name": {"$in": {"test", "test1"}}}
        assert document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_in_no_match_set(self):
        document = Document(metadata={"name": "test"})
        filter = {"metadata.name": {"$in": {"test2", "test3"}}}
        assert not document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_in_match_tuple(self):
        document = Document(metadata={"name": "test"})
        filter = {"metadata.name": {"$in": ("test", "test1")}}
        assert document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_in_no_match_tuple(self):
        document = Document(metadata={"name": "test"})
        filter = {"metadata.name": {"$in": ("test2", "test3")}}
        assert not document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_in_no_match_missing_key(self):
        document = Document(metadata={"name": "test"})
        filter = {"metadata.name1": {"$in": ["test", "test1"]}}
        assert not document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_in_unsupported_type(self):
        document = Document(metadata={"name": "test"})
        filter = {"metadata.name": {"$in": "unsupported"}}
        with pytest.raises(FilterError, match=r"\$in accepts only iterable values like lists, sets and tuples"):
            document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_nin_match_list(self):
        document = Document(metadata={"name": "test"})
        filter = {"metadata.name": {"$nin": ["test1", "test2"]}}
        assert document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_nin_no_match_list(self):
        document = Document(metadata={"name": "test"})
        filter = {"metadata.name": {"$nin": ["test", "test1"]}}
        assert not document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_nin_match_set(self):
        document = Document(metadata={"name": "test"})
        filter = {"metadata.name": {"$nin": {"test1", "test2"}}}
        assert document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_nin_no_match_set(self):
        document = Document(metadata={"name": "test"})
        filter = {"metadata.name": {"$nin": {"test", "test1"}}}
        assert not document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_nin_match_tuple(self):
        document = Document(metadata={"name": "test"})
        filter = {"metadata.name": {"$nin": ("test1", "test2")}}
        assert document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_nin_no_match_tuple(self):
        document = Document(metadata={"name": "test"})
        filter = {"metadata.name": {"$nin": ("test", "test1")}}
        assert not document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_nin_no_match_missing_key(self):
        document = Document(metadata={"name": "test"})
        filter = {"metadata.name1": {"$nin": ["test", "test1"]}}
        assert document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_nin_unsupported_type(self):
        document = Document(metadata={"name": "test"})
        filter = {"metadata.name": {"$nin": "unsupported"}}
        with pytest.raises(FilterError, match=r"\$in accepts only iterable values like lists, sets and tuples"):
            document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_gt_match_int(self):
        document = Document(metadata={"age": 21})
        filter = {"metadata.age": {"$gt": 20}}
        assert document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_gt_no_match_int(self):
        document = Document(metadata={"age": 19})
        filter = {"metadata.age": {"$gt": 20}}
        assert not document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_gt_match_float(self):
        document = Document(metadata={"number": 90.5})
        filter = {"metadata.number": {"$gt": 90.0}}
        assert document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_gt_no_match_float(self):
        document = Document(metadata={"number": 89.5})
        filter = {"metadata.number": {"$gt": 90.0}}
        assert not document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_gt_match_np_number(self):
        document = Document(metadata={"value": np.float64(7.5)})
        filter = {"metadata.value": {"$gt": np.float64(7.0)}}
        assert document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_gt_no_match_np_number(self):
        document = Document(metadata={"value": np.float64(6.5)})
        filter = {"metadata.value": {"$gt": np.float64(7.0)}}
        assert not document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_gt_match_date_string(self):
        document = Document(metadata={"date": "2022-01-02"})
        filter = {"metadata.date": {"$gt": "2022-01-01"}}
        assert document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_gt_no_match_date_string(self):
        document = Document(metadata={"date": "2022-01-01"})
        filter = {"metadata.date": {"$gt": "2022-01-01"}}
        assert not document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_gt_no_match_missing_key(self):
        document = Document(metadata={"age": 21})
        filter = {"metadata.age1": {"$gt": 20}}
        assert not document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_gt_unsupported_type(self):
        document = Document(metadata={"age": 21})
        filter = {"metadata.age": {"$gt": "unsupported"}}
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
        document = Document(metadata={"age": 21})
        filter_1 = {"metadata.age": {"$gte": 21}}
        filter_2 = {"metadata.age": {"$gte": 20}}
        assert document_matches_filter(filter_1, document)
        assert document_matches_filter(filter_2, document)

    @pytest.mark.unit
    def test_gte_no_match_int(self):
        document = Document(metadata={"age": 20})
        filter = {"metadata.age": {"$gte": 21}}
        assert not document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_gte_match_float(self):
        document = Document(metadata={"number": 90.5})
        filter_1 = {"metadata.number": {"$gte": 90.5}}
        filter_2 = {"metadata.number": {"$gte": 90.4}}
        assert document_matches_filter(filter_1, document)
        assert document_matches_filter(filter_2, document)

    @pytest.mark.unit
    def test_gte_no_match_float(self):
        document = Document(metadata={"number": 90.4})
        filter = {"metadata.number": {"$gte": 90.5}}
        assert not document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_gte_match_np_number(self):
        document = Document(metadata={"value": np.float64(7.5)})
        filter_1 = {"metadata.value": {"$gte": np.float64(7.5)}}
        filter_2 = {"metadata.value": {"$gte": np.float64(7.4)}}
        assert document_matches_filter(filter_1, document)
        assert document_matches_filter(filter_2, document)

    @pytest.mark.unit
    def test_gte_no_match_np_number(self):
        document = Document(metadata={"value": np.float64(7.4)})
        filter = {"metadata.value": {"$gte": np.float64(7.5)}}
        assert not document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_gte_match_date_string(self):
        document = Document(metadata={"date": "2022-01-02"})
        filter_1 = {"metadata.date": {"$gte": "2022-01-02"}}
        filter_2 = {"metadata.date": {"$gte": "2022-01-01"}}
        assert document_matches_filter(filter_1, document)
        assert document_matches_filter(filter_2, document)

    @pytest.mark.unit
    def test_gte_no_match_date_string(self):
        document = Document(metadata={"date": "2022-01-01"})
        filter = {"metadata.date": {"$gte": "2022-01-02"}}
        assert not document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_gte_unsupported_type(self):
        document = Document(metadata={"age": 21})
        filter = {"metadata.age": {"$gte": "unsupported"}}
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
        document = Document(metadata={"age": 19})
        filter = {"metadata.age": {"$lt": 20}}
        assert document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_lt_no_match_int(self):
        document = Document(metadata={"age": 20})
        filter = {"metadata.age": {"$lt": 20}}
        assert not document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_lt_match_float(self):
        document = Document(metadata={"number": 89.9})
        filter = {"metadata.number": {"$lt": 90.0}}
        assert document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_lt_no_match_float(self):
        document = Document(metadata={"number": 90.0})
        filter = {"metadata.number": {"$lt": 90.0}}
        assert not document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_lt_match_np_number(self):
        document = Document(metadata={"value": np.float64(6.9)})
        filter = {"metadata.value": {"$lt": np.float64(7.0)}}
        assert document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_lt_no_match_np_number(self):
        document = Document(metadata={"value": np.float64(7.0)})
        filter = {"metadata.value": {"$lt": np.float64(7.0)}}
        assert not document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_lt_match_date_string(self):
        document = Document(metadata={"date": "2022-01-01"})
        filter = {"metadata.date": {"$lt": "2022-01-02"}}
        assert document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_lt_no_match_date_string(self):
        document = Document(metadata={"date": "2022-01-02"})
        filter = {"metadata.date": {"$lt": "2022-01-02"}}
        assert not document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_lt_unsupported_type(self):
        document = Document(metadata={"age": 21})
        filter = {"metadata.age": {"$lt": "unsupported"}}
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
        document = Document(metadata={"age": 21})
        filter_1 = {"metadata.age": {"$lte": 21}}
        filter_2 = {"metadata.age": {"$lte": 20}}
        assert not document_matches_filter(filter_2, document)
        assert document_matches_filter(filter_1, document)

    @pytest.mark.unit
    def test_lte_no_match_int(self):
        document = Document(metadata={"age": 22})
        filter = {"metadata.age": {"$lte": 21}}
        assert not document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_lte_match_float(self):
        document = Document(metadata={"number": 90.5})
        filter_1 = {"metadata.number": {"$lte": 90.5}}
        filter_2 = {"metadata.number": {"$lte": 90.4}}
        assert not document_matches_filter(filter_2, document)
        assert document_matches_filter(filter_1, document)

    @pytest.mark.unit
    def test_lte_no_match_float(self):
        document = Document(metadata={"number": 90.6})
        filter = {"metadata.number": {"$lte": 90.5}}
        assert not document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_lte_match_np_number(self):
        document = Document(metadata={"value": np.float64(7.5)})
        filter_1 = {"metadata.value": {"$lte": np.float64(7.5)}}
        filter_2 = {"metadata.value": {"$lte": np.float64(7.4)}}
        assert not document_matches_filter(filter_2, document)
        assert document_matches_filter(filter_1, document)

    @pytest.mark.unit
    def test_lte_no_match_np_number(self):
        document = Document(metadata={"value": np.float64(7.6)})
        filter = {"metadata.value": {"$lte": np.float64(7.5)}}
        assert not document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_lte_match_date_string(self):
        document = Document(metadata={"date": "2022-01-02"})
        filter_1 = {"metadata.date": {"$lte": "2022-01-02"}}
        filter_2 = {"metadata.date": {"$lte": "2022-01-01"}}
        assert not document_matches_filter(filter_2, document)
        assert document_matches_filter(filter_1, document)

    @pytest.mark.unit
    def test_lte_no_match_date_string(self):
        document = Document(metadata={"date": "2022-01-03"})
        filter = {"metadata.date": {"$lte": "2022-01-02"}}
        assert not document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_lte_unsupported_type(self):
        document = Document(metadata={"age": 21})
        filter = {"metadata.age": {"$lte": "unsupported"}}
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
        document = Document(metadata={"age": 21, "name": "John"})
        filter = {"metadata.age": {"$gt": 18}, "metadata.name": "John"}
        assert document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_explicit_and(self):
        document = Document(metadata={"age": 21})
        filter = {"metadata.age": {"$and": {"$gt": 18}, "$lt": 25}}
        assert document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_or(self):
        document = Document(metadata={"age": 26})
        filter = {"metadata.age": {"$or": [{"$gt": 18}, {"$lt": 25}]}}
        assert document_matches_filter(filter, document)

    @pytest.mark.unit
    def test_not(self):
        document = Document(metadata={"age": 17})
        filter = {"metadata.age": {"$not": {"$gt": 18}}}
        assert document_matches_filter(filter, document)
