import pytest

from haystack.preview import Document, ComponentError
from haystack.preview.components.samplers.top_p import TopPSampler


class TestTopPSampler:
    @pytest.mark.unit
    def test_to_dict(self):
        component = TopPSampler()
        data = component.to_dict()
        assert data == {"type": "TopPSampler", "init_parameters": {"top_p": 1.0, "score_field": None}}

    @pytest.mark.unit
    def test_to_dict_with_custom_init_parameters(self):
        component = TopPSampler(top_p=0.92)
        data = component.to_dict()
        assert data == {"type": "TopPSampler", "init_parameters": {"top_p": 0.92, "score_field": None}}

    @pytest.mark.unit
    def test_from_dict(self):
        data = {"type": "TopPSampler", "init_parameters": {"top_p": 0.9, "score_field": None}}
        component = TopPSampler.from_dict(data)
        assert component.top_p == 0.9

    @pytest.mark.unit
    def test_run_scores_from_metadata(self):
        """
        Test if the component runs correctly with scores already in the metadata.
        """
        sampler = TopPSampler(top_p=0.95, score_field="similarity_score")
        docs = [
            Document(text="Berlin", metadata={"similarity_score": -10.6}),
            Document(text="Belgrade", metadata={"similarity_score": -8.9}),
            Document(text="Sarajevo", metadata={"similarity_score": -4.6}),
        ]
        output = sampler.run(documents=docs)
        docs = output["documents"]
        assert len(docs) == 1
        assert docs[0].text == "Sarajevo"

    @pytest.mark.unit
    def test_run_scores(self):
        """
        Test if the component runs correctly with scores in the Document score field.
        """
        sampler = TopPSampler(top_p=0.95)
        docs = [
            Document(text="Berlin", score=-10.6),
            Document(text="Belgrade", score=-8.9),
            Document(text="Sarajevo", score=-4.6),
        ]
        output = sampler.run(documents=docs)
        docs = output["documents"]
        assert len(docs) == 1
        assert docs[0].text == "Sarajevo"

    #  Returns an empty list if no documents are provided
    @pytest.mark.unit
    def test_returns_empty_list_if_no_documents_are_provided(self):
        sampler = TopPSampler()
        output = sampler.run(documents=[])
        assert output["documents"] == []

    @pytest.mark.unit
    def test_run_scores_no_metadata_present(self):
        """
        Test if the component runs correctly with scores missing from the metadata yet being specified in the
        score_field.
        """
        sampler = TopPSampler(top_p=0.95, score_field="similarity_score")
        docs = [
            Document(text="Berlin", score=-10.6),
            Document(text="Belgrade", score=-8.9),
            Document(text="Sarajevo", score=-4.6),
        ]
        with pytest.raises(ComponentError, match="Score field 'similarity_score' not found"):
            sampler.run(documents=docs)
