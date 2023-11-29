import random

import pytest

from haystack import Document, ComponentError
from haystack.components.samplers.top_p import TopPSampler


class TestTopPSampler:
    def test_run_scores_from_metadata(self):
        """
        Test if the component runs correctly with scores already in the metadata.
        """
        sampler = TopPSampler(top_p=0.95, score_field="similarity_score")
        docs = [
            Document(content="Berlin", meta={"similarity_score": -10.6}),
            Document(content="Belgrade", meta={"similarity_score": -8.9}),
            Document(content="Sarajevo", meta={"similarity_score": -4.6}),
        ]
        output = sampler.run(documents=docs)
        docs = output["documents"]
        assert len(docs) == 1
        assert docs[0].content == "Sarajevo"

    def test_run_scores(self):
        """
        Test if the component runs correctly with scores in the Document score field.
        """
        sampler = TopPSampler(top_p=0.99)
        docs = [
            Document(content="Berlin", score=-10.6),
            Document(content="Belgrade", score=-8.9),
            Document(content="Sarajevo", score=-4.6),
        ]

        random.shuffle(docs)
        sorted_scores = sorted([doc.score for doc in docs], reverse=True)

        # top_p = 0.99 will get the top 1 document
        output = sampler.run(documents=docs)
        docs_filtered = output["documents"]
        assert len(docs_filtered) == 1
        assert docs_filtered[0].content == "Sarajevo"

        assert [doc.score for doc in docs_filtered] == sorted_scores[:1]

    def test_run_scores_top_p_1(self):
        """
        Test if the component runs correctly top_p=1.
        """
        sampler = TopPSampler(top_p=1.0)
        docs = [
            Document(content="Berlin", score=-10.6),
            Document(content="Belgrade", score=-8.9),
            Document(content="Sarajevo", score=-4.6),
        ]

        random.shuffle(docs)
        output = sampler.run(documents=docs)
        docs_filtered = output["documents"]
        assert len(docs_filtered) == len(docs)
        assert docs_filtered[0].content == "Sarajevo"

        assert [doc.score for doc in docs_filtered] == sorted([doc.score for doc in docs], reverse=True)

    #  Returns an empty list if no documents are provided

    def test_returns_empty_list_if_no_documents_are_provided(self):
        sampler = TopPSampler()
        output = sampler.run(documents=[])
        assert output["documents"] == []

    def test_run_scores_no_metadata_present(self):
        """
        Test if the component runs correctly with scores missing from the metadata yet being specified in the
        score_field.
        """
        sampler = TopPSampler(top_p=0.95, score_field="similarity_score")
        docs = [
            Document(content="Berlin", score=-10.6),
            Document(content="Belgrade", score=-8.9),
            Document(content="Sarajevo", score=-4.6),
        ]
        with pytest.raises(ComponentError, match="Score field 'similarity_score' not found"):
            sampler.run(documents=docs)
