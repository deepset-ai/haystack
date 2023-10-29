import pytest

from haystack.preview import Document, ComponentError
from haystack.preview.components.rankers.meta_field import MetaFieldRanker


class TestMetaFieldRanker:
    @pytest.mark.unit
    def test_to_dict(self):
        component = MetaFieldRanker(metadata_field="rating")
        data = component.to_dict()
        assert data == {
            "type": "MetaFieldRanker",
            "init_parameters": {
                "metadata_field": "rating",
                "weight": 1.0,
                "top_k": None,
                "ranking_mode": "reciprocal_rank_fusion",
            },
        }

    @pytest.mark.unit
    def test_to_dict_with_custom_init_parameters(self):
        component = MetaFieldRanker(metadata_field="rating", weight=0.5, top_k=5, ranking_mode="linear_score")
        data = component.to_dict()
        assert data == {
            "type": "MetaFieldRanker",
            "init_parameters": {"metadata_field": "rating", "weight": 0.5, "top_k": 5, "ranking_mode": "linear_score"},
        }

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "query, metafield_values, expected_first_value", [("", [1.3, 0.7, 2.1], 2.1), ("", [1, 5, 8], 8)]
    )
    def test_run(self, query, metafield_values, expected_first_value):
        """
        Test if the component ranks documents correctly.
        """
        ranker = MetaFieldRanker(metadata_field="rating")
        docs_before = [Document(text="abc", metadata={"rating": value}) for value in metafield_values]

        output = ranker.run(query=query, documents=docs_before)
        docs_after = output["documents"]

        assert len(docs_after) == 3
        assert docs_after[0].metadata["rating"] == expected_first_value

        sorted_scores = sorted([doc.metadata["rating"] for doc in docs_after], reverse=True)
        assert [doc.metadata["rating"] for doc in docs_after] == sorted_scores

    @pytest.mark.integration
    def test_returns_empty_list_if_no_documents_are_provided(self):
        ranker = MetaFieldRanker(metadata_field="rating")
        output = ranker.run(query="", documents=[])
        docs_after = output["documents"]
        assert docs_after == []

    @pytest.mark.integration
    def test_raises_component_error_if_metadata_not_found(self):
        ranker = MetaFieldRanker(metadata_field="rating")
        docs_before = [Document(text="abc", metadata={"wrong_field": 1.3})]
        with pytest.raises(ComponentError):
            ranker.run(query="", documents=docs_before)

    @pytest.mark.integration
    def test_raises_component_error_if_wrong_ranking_mode(self):
        with pytest.raises(ComponentError):
            MetaFieldRanker(metadata_field="rating", ranking_mode="wrong_mode")

    @pytest.mark.integration
    @pytest.mark.parametrize("score", [-1, 2, 1.3, 2.1])
    def test_raises_component_error_if_wrong_weight(self, score):
        with pytest.raises(ComponentError):
            MetaFieldRanker(metadata_field="rating", weight=score)

    @pytest.mark.integration
    def test_linear_score(self):
        ranker = MetaFieldRanker(metadata_field="rating", ranking_mode="linear_score", weight=0.5)
        docs_before = [
            Document(text="abc", metadata={"rating": 1.3}, score=0.3),
            Document(text="abc", metadata={"rating": 0.7}, score=0.4),
            Document(text="abc", metadata={"rating": 2.1}, score=0.6),
        ]
        output = ranker.run(query="", documents=docs_before)
        docs_after = output["documents"]
        assert docs_after[0].score == 0.8

    @pytest.mark.integration
    def test_reciprocal_rank_fusion(self):
        ranker = MetaFieldRanker(metadata_field="rating", ranking_mode="reciprocal_rank_fusion", weight=0.5)
        docs_before = [
            Document(text="abc", metadata={"rating": 1.3}, score=0.3),
            Document(text="abc", metadata={"rating": 0.7}, score=0.4),
            Document(text="abc", metadata={"rating": 2.1}, score=0.6),
        ]
        output = ranker.run(query="", documents=docs_before)
        docs_after = output["documents"]
        assert docs_after[0].score == 0.01626123744050767
