# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import random
from typing import List

import pytest
from haystack import Document
from haystack.components.samplers.top_p import TopPSampler


@pytest.fixture
def documents_with_score_field() -> List[Document]:
    return [
        Document(content="Sarajevo", meta={"similarity_score": 0.7}),
        Document(content="Belgrade", meta={"similarity_score": 0.01}),
        Document(content="Berlin", meta={"similarity_score": 0.001}),
    ]


@pytest.fixture
def documents_with_score() -> List[Document]:
    return [
        Document(content="Sarajevo", score=0.7),
        Document(content="Belgrade", score=0.01),
        Document(content="Berlin", score=0.001),
    ]


class TestTopPSampler:
    def test_init_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            TopPSampler(top_p=2.0)

    def test_run_raises_value_error(self, documents_with_score: List[Document]) -> None:
        sampler = TopPSampler(top_p=0.95)
        with pytest.raises(ValueError):
            sampler.run(documents=documents_with_score, top_p=2.0)

    def test_run_score_field(self, documents_with_score_field: List[Document]) -> None:
        sampler = TopPSampler(top_p=0.95, score_field="similarity_score")
        docs = documents_with_score_field
        output = sampler.run(documents=docs)
        docs = output["documents"]
        assert len(docs) == 2
        assert docs[0].content == "Sarajevo"
        assert docs[1].content == "Belgrade"

    def test_run_score_field_missing_scores(self, caplog: pytest.LogCaptureFixture) -> None:
        sampler = TopPSampler(top_p=1.0, score_field="similarity_score")
        docs = [
            Document(content="Sarajevo", meta={"similarity_score": 0.7}),
            Document(content="Belgrade", meta={"similarity_score": 0.01}),
            Document(content="Berlin", meta={"similarity_score": None}),
        ]
        output = sampler.run(documents=docs)
        docs = output["documents"]
        assert len(docs) == 2
        assert docs[0].content == "Sarajevo"
        assert docs[1].content == "Belgrade"
        assert "Score field" in caplog.text

    def test_run(self, documents_with_score: List[Document]) -> None:
        sampler = TopPSampler(top_p=0.99)
        docs = documents_with_score
        random.shuffle(docs)
        sorted_scores = sorted([doc.score for doc in docs], reverse=True)

        # top_p = 0.99 will get the top 1 document
        output = sampler.run(documents=docs)
        docs_filtered = output["documents"]
        assert len(docs_filtered) == 2
        assert docs_filtered[0].content == "Sarajevo"
        assert docs_filtered[1].content == "Belgrade"

        assert [doc.score for doc in docs_filtered] == sorted_scores[:2]

    def test_run_top_p_1(self, documents_with_score: List[Document]) -> None:
        sampler = TopPSampler(top_p=1.0)
        docs = documents_with_score
        random.shuffle(docs)
        output = sampler.run(documents=docs)
        docs_filtered = output["documents"]
        assert len(docs_filtered) == len(docs)
        assert docs_filtered[0].content == "Sarajevo"
        assert [doc.score for doc in docs_filtered] == sorted([doc.score for doc in docs], reverse=True)

    def test_run_top_p_0(self, caplog: pytest.LogCaptureFixture, documents_with_score: List[Document]) -> None:
        sampler = TopPSampler(top_p=0.0)
        docs = documents_with_score
        output = sampler.run(documents=docs)
        docs = output["documents"]
        assert len(docs) == 1
        assert docs[0].content == "Sarajevo"
        assert "Top-p sampling with p=" in caplog.text

    def test_run_returns_empty_list_no_documents(self) -> None:
        sampler = TopPSampler()
        output = sampler.run(documents=[])
        assert output["documents"] == []

    def test_run_no_score_field(self, caplog: pytest.LogCaptureFixture, documents_with_score: List[Document]) -> None:
        sampler = TopPSampler(top_p=0.95, score_field="similarity_score")
        docs = documents_with_score
        output = sampler.run(documents=docs)
        docs = output["documents"]
        assert len(docs) == 3
        assert docs[0].content == "Sarajevo"
        assert "Score field 'similarity_score' not found" in caplog.text

    def test_run_missing_scores(self, caplog: pytest.LogCaptureFixture) -> None:
        sampler = TopPSampler(top_p=0.95)
        docs = [
            Document(content="Sarajevo", score=0.7),
            Document(content="Belgrade", score=0.01),
            Document(content="Berlin", score=None),
        ]
        output = sampler.run(documents=docs)
        docs = output["documents"]
        assert len(docs) == 1
        assert docs[0].content == "Sarajevo"
        assert "Ensure all documents have a valid score value" in caplog.text

    def test_run_min_top_k(self, documents_with_score: List[Document]) -> None:
        sampler = TopPSampler(min_top_k=2, top_p=0.2)
        docs = documents_with_score
        output = sampler.run(documents=docs)
        docs = output["documents"]
        assert len(docs) == 2
        assert docs[0].content == "Sarajevo"
        assert docs[1].content == "Belgrade"
