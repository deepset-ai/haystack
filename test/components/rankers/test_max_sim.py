# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import logging

import pytest

from haystack import Document
from haystack.components.rankers import MaxSimRanker
from haystack.core.serialization import component_from_dict, component_to_dict


class TestInitializationSerde:
    def test_init(self):
        assert MaxSimRanker().top_k == 10
        assert MaxSimRanker(top_k=2).top_k == 2

    def test_init_non_positive_top_k(self):
        with pytest.raises(ValueError, match=r"^top_k must be > 0, but got 0$"):
            MaxSimRanker(top_k=0)

    def test_serde(self):
        ranker = MaxSimRanker(top_k=2)
        serialized = component_to_dict(obj=ranker, name="ranker")
        assert serialized == {
            "type": "haystack.components.rankers.max_sim.MaxSimRanker",
            "init_parameters": {"top_k": 2},
        }

        restored = component_from_dict(cls=MaxSimRanker, data=serialized, name="ranker")

        assert isinstance(restored, MaxSimRanker)
        assert restored.top_k == 2


class TestRun:
    def test_run(self, caplog):
        ranker = MaxSimRanker(top_k=2)
        documents = [
            Document(id="partial", multivector_embedding=[[0.5, 0.5]], score=0.9),
            Document(id="negative", multivector_embedding=[[-1.0, -2.0]]),
            Document(id="missing_embedding"),
            Document(id="best", multivector_embedding=[[1.0, 0.0], [0.0, 2.0]], score=0.1),
        ]
        original_documents = [document.to_dict() for document in documents]

        with caplog.at_level(logging.WARNING):
            result = ranker.run(query_multivector_embedding=[[1.0, 0.0], [1.0, 1.0]], documents=documents)

        assert list(result) == ["documents"]
        assert [document.id for document in result["documents"]] == ["best", "partial"]
        assert [document.score for document in result["documents"]] == pytest.approx([3.0, 1.5])
        assert [document.to_dict() for document in documents] == original_documents
        assert "Skipping documents without multivector embeddings: ['missing_embedding']" in caplog.text

    def test_run_top_k_override(self):
        ranker = MaxSimRanker(top_k=1)
        documents = [
            Document(id="lowest", multivector_embedding=[[1.0]]),
            Document(id="highest", multivector_embedding=[[3.0]]),
            Document(id="middle", multivector_embedding=[[2.0]]),
        ]

        result = ranker.run(query_multivector_embedding=[[1.0]], documents=documents, top_k=2)

        assert [document.id for document in result["documents"]] == ["highest", "middle"]
        assert ranker.top_k == 1

    def test_run_no_documents(self):
        ranker = MaxSimRanker()

        assert ranker.run(query_multivector_embedding=[[1.0, 0.0]], documents=[]) == {"documents": []}
        assert ranker.run(query_multivector_embedding=[[1.0, 0.0]], documents=[Document()]) == {"documents": []}

    def test_run_non_positive_top_k(self):
        ranker = MaxSimRanker()
        documents = [Document(multivector_embedding=[[1.0, 0.0]])]

        with pytest.raises(ValueError, match=r"^top_k must be > 0, but got 0$"):
            ranker.run(query_multivector_embedding=[[1.0, 0.0]], documents=documents, top_k=0)

    @pytest.mark.parametrize(
        "query_multivector_embedding", [[], [[]], [1.0, 2.0], [[float("nan")]], [[1.0], [1.0, 2.0]]]
    )
    def test_run_invalid_query_embedding(self, query_multivector_embedding):
        ranker = MaxSimRanker()
        documents = [Document(multivector_embedding=[[1.0]])]

        with pytest.raises(
            ValueError, match=r"^query_multivector_embedding must be a non-empty matrix of finite numbers\.$"
        ):
            ranker.run(query_multivector_embedding=query_multivector_embedding, documents=documents)

    @pytest.mark.parametrize("multivector_embedding", [[], [[]]])
    def test_run_invalid_document_embedding(self, multivector_embedding):
        ranker = MaxSimRanker()
        documents = [Document(id="invalid", multivector_embedding=multivector_embedding)]

        with pytest.raises(
            ValueError,
            match=r"^multivector_embedding of document 'invalid' must be a non-empty matrix of finite numbers\.$",
        ):
            ranker.run(query_multivector_embedding=[[1.0, 0.0]], documents=documents)

    def test_run_query_documents_embedding_mismatch(self):
        ranker = MaxSimRanker()
        documents = [Document(id="invalid", multivector_embedding=[[1.0, 0.0, 0.0]])]

        with pytest.raises(
            ValueError,
            match=(
                r"^multivector_embedding of document 'invalid' has dimension 3, "
                r"but query_multivector_embedding has dimension 2\.$"
            ),
        ):
            ranker.run(query_multivector_embedding=[[1.0, 0.0]], documents=documents)
