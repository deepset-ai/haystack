import pytest

from haystack.preview import Document, ComponentError
from haystack.preview.components.rankers.similarity import SimilarityRanker


class TestSimilarityRanker:
    @pytest.mark.unit
    def test_to_dict(self):
        component = SimilarityRanker(model_name_or_path="cross-encoder/ms-marco-MiniLM-L-6-v2")
        data = component.to_dict()
        assert data == {
            "type": "SimilarityRanker",
            "init_parameters": {
                "device": "cpu",
                "top_k": 10,
                "model_name_or_path": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            },
        }

    @pytest.mark.unit
    def test_to_dict_with_custom_init_parameters(self):
        component = SimilarityRanker()
        data = component.to_dict()
        assert data == {
            "type": "SimilarityRanker",
            "init_parameters": {
                "device": "cpu",
                "top_k": 10,
                "model_name_or_path": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            },
        }

    @pytest.mark.integration
    def test_from_dict(self):
        data = {
            "type": "SimilarityRanker",
            "init_parameters": {
                "device": "cpu",
                "top_k": 10,
                "model_name_or_path": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            },
        }
        component = SimilarityRanker.from_dict(data)
        assert component.model_name_or_path == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "query,docs_before_texts,expected_first_text",
        [
            ("City in Bosnia and Herzegovina", ["Berlin", "Belgrade", "Sarajevo"], "Sarajevo"),
            ("Machine learning", ["Python", "Bakery in Paris", "Tesla Giga Berlin"], "Python"),
            ("Cubist movement", ["Nirvana", "Pablo Picasso", "Coffee"], "Pablo Picasso"),
        ],
    )
    def test_run(self, query, docs_before_texts, expected_first_text):
        """
        Test if the component ranks documents correctly.
        """
        ranker = SimilarityRanker(model_name_or_path="cross-encoder/ms-marco-MiniLM-L-6-v2")
        ranker.warm_up()
        docs_before = [Document(text=text) for text in docs_before_texts]
        output = ranker.run(query=query, documents=docs_before)
        docs_after = output["documents"]

        assert len(docs_after) == 3
        assert docs_after[0].text == expected_first_text

        sorted_scores = sorted([doc.score for doc in docs_after], reverse=True)
        assert [doc.score for doc in docs_after] == sorted_scores

    #  Returns an empty list if no documents are provided
    @pytest.mark.integration
    def test_returns_empty_list_if_no_documents_are_provided(self):
        sampler = SimilarityRanker()
        sampler.warm_up()
        output = sampler.run(query="City in Germany", documents=[])
        assert output["documents"] == []

    #  Raises ComponentError if model is not warmed up
    @pytest.mark.integration
    def test_raises_component_error_if_model_not_warmed_up(self):
        sampler = SimilarityRanker()

        with pytest.raises(ComponentError):
            sampler.run(query="query", documents=[Document(text="document")])

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "query,docs_before_texts,expected_first_text",
        [
            ("City in Bosnia and Herzegovina", ["Berlin", "Belgrade", "Sarajevo"], "Sarajevo"),
            ("Machine learning", ["Python", "Bakery in Paris", "Tesla Giga Berlin"], "Python"),
            ("Cubist movement", ["Nirvana", "Pablo Picasso", "Coffee"], "Pablo Picasso"),
        ],
    )
    def test_run_top_k(self, query, docs_before_texts, expected_first_text):
        """
        Test if the component ranks documents correctly with a custom top_k.
        """
        ranker = SimilarityRanker(model_name_or_path="cross-encoder/ms-marco-MiniLM-L-6-v2", top_k=2)
        ranker.warm_up()
        docs_before = [Document(text=text) for text in docs_before_texts]
        output = ranker.run(query=query, documents=docs_before)
        docs_after = output["documents"]

        assert len(docs_after) == 2
        assert docs_after[0].text == expected_first_text

        sorted_scores = sorted([doc.score for doc in docs_after], reverse=True)
        assert [doc.score for doc in docs_after] == sorted_scores
