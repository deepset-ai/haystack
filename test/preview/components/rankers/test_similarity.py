import pytest

from haystack.preview import Document, ComponentError
from haystack.preview.components.rankers.similarity import SimilarityRanker


class TestSimilarityRanker:
    @pytest.mark.integration
    def test_to_dict(self):
        component = SimilarityRanker(model_name_or_path="cross-encoder/ms-marco-MiniLM-L-6-v2")
        data = component.to_dict()
        assert data == {
            "type": "SimilarityRanker",
            "init_parameters": {"device": "cpu", "model_name_or_path": "cross-encoder/ms-marco-MiniLM-L-6-v2"},
        }

    @pytest.mark.integration
    def test_to_dict_with_custom_init_parameters(self):
        component = SimilarityRanker()
        data = component.to_dict()
        assert data == {
            "type": "SimilarityRanker",
            "init_parameters": {"device": "cpu", "model_name_or_path": "cross-encoder/ms-marco-MiniLM-L-6-v2"},
        }

    @pytest.mark.integration
    def test_from_dict(self):
        data = {
            "type": "SimilarityRanker",
            "init_parameters": {"device": "cpu", "model_name_or_path": "cross-encoder/ms-marco-MiniLM-L-6-v2"},
        }
        component = SimilarityRanker.from_dict(data)
        assert component.model_name_or_path == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    @pytest.mark.integration
    def test_run(self):
        """
        Test if the component runs correctly.
        """
        sampler = SimilarityRanker(model_name_or_path="cross-encoder/ms-marco-MiniLM-L-6-v2")
        sampler.warm_up()
        docs = [Document(text="Berlin"), Document(text="Belgrade"), Document(text="Sarajevo")]
        query = "City in Bosnia and Herzegovina"
        output = sampler.run(query=query, documents=docs)
        docs = output["documents"]
        assert len(docs) == 3
        assert docs[0].text == "Sarajevo"

        # another test to make sure the first one was not a fluke
        docs = [Document(text="Python"), Document(text="Bakery in Paris"), Document(text="Tesla Giga Berlin")]
        query = "Programming language usually used for machine learning?"
        output = sampler.run(query=query, documents=docs)
        docs = output["documents"]
        assert len(docs) == 3
        assert docs[0].text == "Python"

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
