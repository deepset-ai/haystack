import sys
from typing import List

import pytest

from haystack.schema import Document
from haystack.nodes.answer_generator import Seq2SeqGenerator
from haystack.document_stores import InMemoryDocumentStore
from haystack.pipelines import GenerativeQAPipeline
from haystack.nodes import DensePassageRetriever


@pytest.fixture
def docs():
    return [
        Document(content="The capital of Germany is the city state of Berlin."),
        Document(content="Berlin is the capital and largest city of Germany by both area and population."),
    ]


def test_lfqa_pipeline(docs):
    document_store = InMemoryDocumentStore(use_bm25=True)
    retriever = DensePassageRetriever(
        document_store=document_store,
        query_embedding_model="vblagoje/dpr-question_encoder-single-lfqa-wiki",
        passage_embedding_model="vblagoje/dpr-ctx_encoder-single-lfqa-wiki",
        use_gpu=False,
        embed_title=True,
    )
    lfqa_generator = Seq2SeqGenerator(model_name_or_path="vblagoje/bart_lfqa", min_length=100, max_length=200)

    document_store.write_documents(docs)
    document_store.update_embeddings(retriever)

    query = "Tell me about Berlin?"
    pipeline = GenerativeQAPipeline(generator=lfqa_generator, retriever=retriever)
    output = pipeline.run(query=query, params={"top_k": 1})
    answers = output["answers"]
    assert len(answers) == 1, answers
    assert "germany" in answers[0].answer.lower()
