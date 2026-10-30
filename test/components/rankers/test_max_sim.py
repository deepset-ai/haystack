# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack import Document, Pipeline
from haystack.components.rankers import MaxSimRanker


def test_scores_ranking_and_top_k_without_mutating_documents():
    documents = [
        Document(id="negative", multi_vector_embedding=[[-1.0, -2.0]]),
        Document(id="best", multi_vector_embedding=[[1.0, 0.0], [0.0, 1.0]]),
        Document(id="partial", multi_vector_embedding=[[0.5, 0.0]]),
    ]
    pipeline = Pipeline()
    pipeline.add_component("ranker", MaxSimRanker(top_k=1))
    pipeline = Pipeline.loads(pipeline.dumps())
    result = pipeline.run(
        {"ranker": {"query_embedding": [[1.0, 0.0], [0.0, 1.0]], "documents": documents, "top_k": 3}}
    )["ranker"]["documents"]
    assert [(doc.id, doc.score) for doc in result] == [("best", 2.0), ("partial", 0.5), ("negative", -3.0)]
    assert all(doc.score is None for doc in documents)
    assert (
        MaxSimRanker(top_k=1).run(query_embedding=[[1.0, 0.0], [0.0, 1.0]], documents=documents)["documents"]
        == result[:1]
    )


def test_empty_documents():
    assert MaxSimRanker().run(query_embedding=[[1.0]], documents=[]) == {"documents": []}


@pytest.mark.parametrize(
    ("embedding", "message"),
    [(None, "must have a non-empty"), ([], "must have a non-empty"), ([[1.0]], "must have the same dimension")],
)
def test_invalid_document_embeddings(embedding, message):
    with pytest.raises(ValueError, match=message):
        MaxSimRanker().run(query_embedding=[[1.0, 0.0]], documents=[Document(multi_vector_embedding=embedding)])


def test_invalid_top_k():
    with pytest.raises(ValueError, match="top_k must be greater than zero"):
        MaxSimRanker(top_k=0)
