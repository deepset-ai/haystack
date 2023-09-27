import pytest

from haystack.preview import Document, ComponentError
from haystack.preview.components.samplers.top_p import TopPSampler


class TestTopP:
    @pytest.mark.integration
    def test_to_dict(self):
        component = TopPSampler(model_name_or_path="cross-encoder/ms-marco-MiniLM-L-6-v2")
        data = component.to_dict()
        assert data == {
            "type": "TopPSampler",
            "init_parameters": {
                "top_p": 1.0,
                "score_field": "similarity_score",
                "device": "cpu",
                "model_name_or_path": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            },
        }

    @pytest.mark.integration
    def test_to_dict_with_custom_init_parameters(self):
        component = TopPSampler(top_p=0.92)
        data = component.to_dict()
        assert data == {
            "type": "TopPSampler",
            "init_parameters": {
                "top_p": 0.92,
                "score_field": "similarity_score",
                "device": "cpu",
                "model_name_or_path": None,
            },
        }

    @pytest.mark.integration
    def test_from_dict(self):
        data = {
            "type": "TopPSampler",
            "init_parameters": {
                "top_p": 0.9,
                "score_field": "similarity_score",
                "device": "cpu",
                "model_name_or_path": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            },
        }
        component = TopPSampler.from_dict(data)
        assert component.model_name_or_path == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert component.top_p == 0.9

    @pytest.mark.integration
    def test_run(self):
        """
        Test if the component runs correctly.
        """
        sampler = TopPSampler(model_name_or_path="cross-encoder/ms-marco-MiniLM-L-6-v2", top_p=0.95)
        sampler.warm_up()
        docs = [Document(text="Berlin"), Document(text="Belgrade"), Document(text="Sarajevo")]
        query = "City in Bosnia and Herzegovina"
        output = sampler.run(query=query, documents=docs)
        docs = output["documents"]
        assert len(docs) == 1
        assert docs[0].text == "Sarajevo"

    @pytest.mark.integration
    def test_run_scores(self):
        """
        Test if the component runs correctly with scores already in the metadata.
        """
        sampler = TopPSampler(top_p=0.95)
        sampler.warm_up()
        docs = [
            Document(text="Berlin", metadata={"similarity_score": -10.6}),
            Document(text="Belgrade", metadata={"similarity_score": -8.9}),
            Document(text="Sarajevo", metadata={"similarity_score": -4.6}),
        ]
        query = "City in Bosnia and Herzegovina"
        output = sampler.run(query=query, documents=docs)
        docs = output["documents"]
        assert len(docs) == 1
        assert docs[0].text == "Sarajevo"

    #  Returns an empty list if no documents are provided
    @pytest.mark.integration
    def test_returns_empty_list_if_no_documents_are_provided(self):
        sampler = TopPSampler()
        sampler.warm_up()
        output = sampler.run(query="City in Germany", documents=[])
        assert output["documents"] == []

    #  Raises ComponentError if model is not warmed up
    @pytest.mark.integration
    def test_raises_component_error_if_model_not_warmed_up(self):
        sampler = TopPSampler()

        with pytest.raises(ComponentError):
            sampler.run(query="query", documents=[Document(text="document")])

    #  Raises ComponentError if model_name_or_path is not set and documents do not have scores
    @pytest.mark.integration
    def test_raises_component_error_if_model_name_or_path_not_set_and_documents_do_not_have_scores(self):
        sampler = TopPSampler(model_name_or_path=None)
        sampler.warm_up()
        docs = [Document(text="Paris"), Document(text="Berlin")]
        query = "City in Germany"

        with pytest.raises(ComponentError):
            sampler.run(query=query, documents=docs)

    #  Returns at least one document if top_p is set to 0
    @pytest.mark.integration
    def test_returns_at_least_one_document_if_top_p_is_set_to_0(self):
        sampler = TopPSampler(model_name_or_path="cross-encoder/ms-marco-MiniLM-L-6-v2", top_p=0)
        sampler.warm_up()

        docs = [Document(text="Paris"), Document(text="Berlin")]
        query = "City in Germany"
        output = sampler.run(query=query, documents=docs)
        filtered_docs = output["documents"]

        # Assert that at least one document is returned
        assert len(filtered_docs) >= 1
