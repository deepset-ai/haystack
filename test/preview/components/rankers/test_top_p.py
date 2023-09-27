import pytest

from haystack.preview import Document
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
