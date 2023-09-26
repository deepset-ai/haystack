import pytest

from haystack.preview import Document
from haystack.preview.components.samplers.top_p import TopPSampler
from haystack.preview.lazy_imports import LazyImport

with LazyImport(
    message="Run 'pip install transformers[torch,sentencepiece]==4.32.1 sentence-transformers>=2.2.0'"
) as torch_and_transformers_import:
    from torch import device


class TestTopP:
    torch_and_transformers_import.check()

    @pytest.mark.integration
    def test_to_dict(self):
        component = TopPSampler()
        data = component.to_dict()
        assert data == {
            "type": "TopPSampler",
            "init_parameters": {
                "top_p": 1.0,
                "score_field": "score",
                "devices": [device(type="cpu")],
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
                "score_field": "score",
                "devices": [device(type="cpu")],
                "model_name_or_path": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            },
        }

    @pytest.mark.integration
    def test_from_dict(self):
        data = {
            "type": "TopPSampler",
            "init_parameters": {
                "top_p": 0.9,
                "score_field": "score",
                "devices": [device(type="cpu")],
                "model_name_or_path": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            },
        }
        component = TopPSampler.from_dict(data)
        assert component.model_name_or_path == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert component.top_p == 0.9

    def test_run(self):
        """
        Test if the component runs correctly.
        """
        ranker = TopPSampler(top_p=0.95)
        docs = [Document(text="Sarajevo"), Document(text="Berlin")]
        query = "City in Bosnia and Herzegovina"
        output = ranker.run(query=query, documents=docs)
        docs = output["documents"]
        assert len(docs) == 1
        assert docs[0].text == "Sarajevo"
