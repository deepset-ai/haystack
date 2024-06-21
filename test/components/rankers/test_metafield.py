# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import logging

import pytest

from haystack import Document
from haystack.components.rankers.meta_field import MetaFieldRanker


class TestMetaFieldRanker:
    @pytest.mark.parametrize("meta_field_values, expected_first_value", [([1.3, 0.7, 2.1], 2.1), ([1, 5, 8], 8)])
    def test_run(self, meta_field_values, expected_first_value):
        """
        Test if the component ranks documents correctly.
        """
        ranker = MetaFieldRanker(meta_field="rating")
        docs_before = [Document(content="abc", meta={"rating": value}) for value in meta_field_values]

        output = ranker.run(documents=docs_before)
        docs_after = output["documents"]

        assert len(docs_after) == 3
        assert docs_after[0].meta["rating"] == expected_first_value

        sorted_scores = sorted([doc.meta["rating"] for doc in docs_after], reverse=True)
        assert [doc.meta["rating"] for doc in docs_after] == sorted_scores

    def test_run_with_weight_equal_to_0(self):
        ranker = MetaFieldRanker(meta_field="rating", weight=0.0)
        docs_before = [Document(content="abc", meta={"rating": value}) for value in [1.1, 0.5, 2.3]]
        output = ranker.run(documents=docs_before)
        docs_after = output["documents"]

        assert len(docs_after) == 3
        assert [doc.meta["rating"] for doc in docs_after] == [1.1, 0.5, 2.3]

    def test_run_with_weight_equal_to_1(self):
        ranker = MetaFieldRanker(meta_field="rating", weight=1.0)
        docs_before = [Document(content="abc", meta={"rating": value}) for value in [1.1, 0.5, 2.3]]
        output = ranker.run(documents=docs_before)
        docs_after = output["documents"]

        assert len(docs_after) == 3
        sorted_scores = sorted([doc.meta["rating"] for doc in docs_after], reverse=True)
        assert [doc.meta["rating"] for doc in docs_after] == sorted_scores

    def test_run_with_weight_equal_to_1_passed_in_run_method(self):
        ranker = MetaFieldRanker(meta_field="rating", weight=0.0)
        docs_before = [Document(content="abc", meta={"rating": value}) for value in [1.1, 0.5, 2.3]]
        output = ranker.run(documents=docs_before, weight=1.0)
        docs_after = output["documents"]

        assert len(docs_after) == 3
        sorted_scores = sorted([doc.meta["rating"] for doc in docs_after], reverse=True)
        assert [doc.meta["rating"] for doc in docs_after] == sorted_scores

    def test_sort_order_ascending(self):
        ranker = MetaFieldRanker(meta_field="rating", weight=1.0, sort_order="ascending")
        docs_before = [Document(content="abc", meta={"rating": value}) for value in [1.1, 0.5, 2.3]]
        output = ranker.run(documents=docs_before)
        docs_after = output["documents"]

        assert len(docs_after) == 3
        sorted_scores = sorted([doc.meta["rating"] for doc in docs_after])
        assert [doc.meta["rating"] for doc in docs_after] == sorted_scores

    def test_meta_value_type_float(self):
        ranker = MetaFieldRanker(meta_field="rating", weight=1.0, meta_value_type="float")
        docs_before = [Document(content="abc", meta={"rating": value}) for value in ["1.1", "10.5", "2.3"]]
        docs_after = ranker.run(documents=docs_before)["documents"]
        assert len(docs_after) == 3
        assert [doc.meta["rating"] for doc in docs_after] == ["10.5", "2.3", "1.1"]

    def test_meta_value_type_int(self):
        ranker = MetaFieldRanker(meta_field="rating", weight=1.0, meta_value_type="int")
        docs_before = [Document(content="abc", meta={"rating": value}) for value in ["1", "10", "2"]]
        docs_after = ranker.run(documents=docs_before)["documents"]
        assert len(docs_after) == 3
        assert [doc.meta["rating"] for doc in docs_after] == ["10", "2", "1"]

    def test_meta_value_type_date(self):
        ranker = MetaFieldRanker(meta_field="rating", weight=1.0, meta_value_type="date")
        docs_before = [Document(content="abc", meta={"rating": value}) for value in ["2022-10", "2023-01", "2022-11"]]
        docs_after = ranker.run(documents=docs_before)["documents"]
        assert len(docs_after) == 3
        assert [doc.meta["rating"] for doc in docs_after] == ["2023-01", "2022-11", "2022-10"]

    def test_returns_empty_list_if_no_documents_are_provided(self):
        ranker = MetaFieldRanker(meta_field="rating")
        output = ranker.run(documents=[])
        docs_after = output["documents"]
        assert docs_after == []

    def test_warning_if_meta_not_found(self, caplog):
        ranker = MetaFieldRanker(meta_field="rating")
        docs_before = [Document(id="1", content="abc", meta={"wrong_field": 1.3})]
        with caplog.at_level(logging.WARNING):
            ranker.run(documents=docs_before)
            assert (
                "The parameter <meta_field> is currently set to 'rating', but none of the provided Documents with IDs 1 have this meta key."
                in caplog.text
            )

    def test_warning_if_some_meta_not_found(self, caplog):
        ranker = MetaFieldRanker(meta_field="rating")
        docs_before = [
            Document(id="1", content="abc", meta={"wrong_field": 1.3}),
            Document(id="2", content="def", meta={"rating": 1.3}),
        ]
        with caplog.at_level(logging.WARNING):
            ranker.run(documents=docs_before)
            assert (
                "The parameter <meta_field> is currently set to 'rating' but the Documents with IDs 1 don't have this meta key."
                in caplog.text
            )

    def test_warning_if_unsortable_values(self, caplog):
        ranker = MetaFieldRanker(meta_field="rating")
        docs_before = [
            Document(id="1", content="abc", meta={"rating": 1.3}),
            Document(id="2", content="abc", meta={"rating": "1.2"}),
            Document(id="3", content="abc", meta={"rating": 2.1}),
        ]
        with caplog.at_level(logging.WARNING):
            output = ranker.run(documents=docs_before)
            assert len(output["documents"]) == 3
            assert "Tried to sort Documents with IDs 1,2,3, but got TypeError with the message:" in caplog.text

    def test_warning_if_meta_value_parsing_error(self, caplog):
        ranker = MetaFieldRanker(meta_field="rating", meta_value_type="float")
        docs_before = [
            Document(id="1", content="abc", meta={"rating": "1.3"}),
            Document(id="2", content="abc", meta={"rating": "1.2"}),
            Document(id="3", content="abc", meta={"rating": "not a float"}),
        ]
        with caplog.at_level(logging.WARNING):
            output = ranker.run(documents=docs_before)
            assert len(output["documents"]) == 3
            assert (
                "Tried to parse the meta values of Documents with IDs 1,2,3, but got ValueError with the message:"
                in caplog.text
            )

    def test_warning_meta_value_type_not_all_strings(self, caplog):
        ranker = MetaFieldRanker(meta_field="rating", meta_value_type="float")
        docs_before = [
            Document(id="1", content="abc", meta={"rating": "1.3"}),
            Document(id="2", content="abc", meta={"rating": "1.2"}),
            Document(id="3", content="abc", meta={"rating": 2.1}),
        ]
        with caplog.at_level(logging.WARNING):
            output = ranker.run(documents=docs_before)
            assert len(output["documents"]) == 3
            assert (
                "The parameter <meta_value_type> is currently set to 'float', but not all of meta values in the provided Documents with IDs 1,2,3 are strings."
                in caplog.text
            )

    def test_raises_value_error_if_wrong_ranking_mode(self):
        with pytest.raises(ValueError):
            MetaFieldRanker(meta_field="rating", ranking_mode="wrong_mode")

    def test_raises_value_error_if_wrong_top_k(self):
        with pytest.raises(ValueError):
            MetaFieldRanker(meta_field="rating", top_k=-1)

    @pytest.mark.parametrize("score", [-1, 2, 1.3, 2.1])
    def test_raises_component_error_if_wrong_weight(self, score):
        with pytest.raises(ValueError):
            MetaFieldRanker(meta_field="rating", weight=score)

    def test_raises_value_error_if_wrong_sort_order(self):
        with pytest.raises(ValueError):
            MetaFieldRanker(meta_field="rating", sort_order="wrong_order")

    def test_raises_value_error_if_wrong_missing_meta(self):
        with pytest.raises(ValueError):
            MetaFieldRanker(meta_field="rating", missing_meta="wrong_missing_meta")

    def test_raises_value_error_if_wrong_meta_value_type(self):
        with pytest.raises(ValueError):
            MetaFieldRanker(meta_field="rating", meta_value_type="wrong_type")

    def test_linear_score(self):
        ranker = MetaFieldRanker(meta_field="rating", ranking_mode="linear_score", weight=0.5)
        docs_before = [
            Document(content="abc", meta={"rating": 1.3}, score=0.3),
            Document(content="abc", meta={"rating": 0.7}, score=0.4),
            Document(content="abc", meta={"rating": 2.1}, score=0.6),
        ]
        output = ranker.run(documents=docs_before)
        docs_after = output["documents"]
        assert docs_after[0].score == 0.8

    def test_reciprocal_rank_fusion(self):
        ranker = MetaFieldRanker(meta_field="rating", ranking_mode="reciprocal_rank_fusion", weight=0.5)
        docs_before = [
            Document(content="abc", meta={"rating": 1.3}, score=0.3),
            Document(content="abc", meta={"rating": 0.7}, score=0.4),
            Document(content="abc", meta={"rating": 2.1}, score=0.6),
        ]
        output = ranker.run(documents=docs_before)
        docs_after = output["documents"]
        assert docs_after[0].score == pytest.approx(0.016261, abs=1e-5)

    @pytest.mark.parametrize("score", [-1, 2, 1.3, 2.1])
    def test_linear_score_raises_warning_if_doc_wrong_score(self, score, caplog):
        ranker = MetaFieldRanker(meta_field="rating", ranking_mode="linear_score", weight=0.5)
        docs_before = [
            Document(id="1", content="abc", meta={"rating": 1.3}, score=score),
            Document(id="2", content="abc", meta={"rating": 0.7}, score=0.4),
            Document(id="3", content="abc", meta={"rating": 2.1}, score=0.6),
        ]
        with caplog.at_level(logging.WARNING):
            ranker.run(documents=docs_before)
            assert f"The score {score} for Document 1 is outside the [0,1] range; defaulting to 0" in caplog.text

    def test_linear_score_raises_raises_warning_if_doc_without_score(self, caplog):
        ranker = MetaFieldRanker(meta_field="rating", ranking_mode="linear_score", weight=0.5)
        docs_before = [
            Document(content="abc", meta={"rating": 1.3}),
            Document(content="abc", meta={"rating": 0.7}),
            Document(content="abc", meta={"rating": 2.1}),
        ]

        with caplog.at_level(logging.WARNING):
            ranker.run(documents=docs_before)
            assert "The score wasn't provided; defaulting to 0." in caplog.text

    def test_different_ranking_mode_for_init_vs_run(self):
        ranker = MetaFieldRanker(meta_field="rating", ranking_mode="linear_score", weight=0.5)
        docs_before = [
            Document(content="abc", meta={"rating": 1.3}, score=0.3),
            Document(content="abc", meta={"rating": 0.7}, score=0.4),
            Document(content="abc", meta={"rating": 2.1}, score=0.6),
        ]
        output = ranker.run(documents=docs_before)
        docs_after = output["documents"]
        assert docs_after[0].score == 0.8

        output = ranker.run(documents=docs_before, ranking_mode="reciprocal_rank_fusion")
        docs_after = output["documents"]
        assert docs_after[0].score == pytest.approx(0.016261, abs=1e-5)

    def test_missing_meta_bottom(self):
        ranker = MetaFieldRanker(meta_field="rating", ranking_mode="linear_score", weight=0.5, missing_meta="bottom")
        docs_before = [
            Document(id="1", content="abc", meta={"rating": 1.3}, score=0.6),
            Document(id="2", content="abc", meta={}, score=0.4),
            Document(id="3", content="abc", meta={"rating": 2.1}, score=0.39),
        ]
        output = ranker.run(documents=docs_before)
        docs_after = output["documents"]
        assert len(docs_after) == 3
        assert docs_after[2].id == "2"

    def test_missing_meta_top(self):
        ranker = MetaFieldRanker(meta_field="rating", ranking_mode="linear_score", weight=0.5, missing_meta="top")
        docs_before = [
            Document(id="1", content="abc", meta={"rating": 1.3}, score=0.6),
            Document(id="2", content="abc", meta={}, score=0.59),
            Document(id="3", content="abc", meta={"rating": 2.1}, score=0.4),
        ]
        output = ranker.run(documents=docs_before)
        docs_after = output["documents"]
        assert len(docs_after) == 3
        assert docs_after[0].id == "2"

    def test_missing_meta_drop(self):
        ranker = MetaFieldRanker(meta_field="rating", ranking_mode="linear_score", weight=0.5, missing_meta="drop")
        docs_before = [
            Document(id="1", content="abc", meta={"rating": 1.3}, score=0.6),
            Document(id="2", content="abc", meta={}, score=0.59),
            Document(id="3", content="abc", meta={"rating": 2.1}, score=0.4),
        ]
        output = ranker.run(documents=docs_before)
        docs_after = output["documents"]
        assert len(docs_after) == 2
        assert "2" not in [doc.id for doc in docs_after]
