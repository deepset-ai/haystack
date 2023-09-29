import pytest

from haystack.preview import Document
from haystack.preview.components.samplers.top_p import TopPSampler


class TestTopPSampler:
    @pytest.mark.integration
    def test_to_dict(self):
        component = TopPSampler()
        data = component.to_dict()
        assert data == {"type": "TopPSampler", "init_parameters": {"top_p": 1.0, "score_field": "similarity_score"}}

    @pytest.mark.integration
    def test_to_dict_with_custom_init_parameters(self):
        component = TopPSampler(top_p=0.92)
        data = component.to_dict()
        assert data == {"type": "TopPSampler", "init_parameters": {"top_p": 0.92, "score_field": "similarity_score"}}

    @pytest.mark.integration
    def test_from_dict(self):
        data = {"type": "TopPSampler", "init_parameters": {"top_p": 0.9, "score_field": "similarity_score"}}
        component = TopPSampler.from_dict(data)
        assert component.top_p == 0.9

    @pytest.mark.integration
    def test_run_scores(self):
        """
        Test if the component runs correctly with scores already in the metadata.
        """
        sampler = TopPSampler(top_p=0.95)
        docs = [
            Document(text="Berlin", metadata={"similarity_score": -10.6}),
            Document(text="Belgrade", metadata={"similarity_score": -8.9}),
            Document(text="Sarajevo", metadata={"similarity_score": -4.6}),
        ]
        output = sampler.run(documents=docs)
        docs = output["documents"]
        assert len(docs) == 1
        assert docs[0].text == "Sarajevo"

    #  Returns an empty list if no documents are provided
    @pytest.mark.integration
    def test_returns_empty_list_if_no_documents_are_provided(self):
        sampler = TopPSampler()
        output = sampler.run(documents=[])
        assert output["documents"] == []
