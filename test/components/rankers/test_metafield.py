import pytest
import logging

from haystack import Document
from haystack.components.rankers.meta_field import MetaFieldRanker


class TestMetaFieldRanker:
    def test_to_dict(self):
        component = MetaFieldRanker(meta_field="rating")
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.rankers.meta_field.MetaFieldRanker",
            "init_parameters": {
                "meta_field": "rating",
                "weight": 1.0,
                "top_k": None,
                "ranking_mode": "reciprocal_rank_fusion",
                "infer_type": False,
            },
        }

    def test_to_dict_with_custom_init_parameters(self):
        component = MetaFieldRanker(
            meta_field="rating", weight=0.5, top_k=5, ranking_mode="linear_score", infer_type=True
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.rankers.meta_field.MetaFieldRanker",
            "init_parameters": {
                "meta_field": "rating",
                "weight": 0.5,
                "top_k": 5,
                "ranking_mode": "linear_score",
                "infer_type": True,
            },
        }

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

    def test_sort_order_ascending(self):
        ranker = MetaFieldRanker(meta_field="rating", weight=1.0, sort_order="ascending")
        docs_before = [Document(content="abc", meta={"rating": value}) for value in [1.1, 0.5, 2.3]]
        output = ranker.run(documents=docs_before)
        docs_after = output["documents"]

        assert len(docs_after) == 3
        sorted_scores = sorted([doc.meta["rating"] for doc in docs_after])
        assert [doc.meta["rating"] for doc in docs_after] == sorted_scores

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
        assert docs_after[0].score == 0.01626123744050767

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
