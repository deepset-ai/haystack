import sys
from typing import List

import numpy as np
import pytest

from haystack.schema import Document
from haystack.nodes.answer_generator import Seq2SeqGenerator
from haystack.pipelines import TranslationWrapperPipeline, GenerativeQAPipeline


from conftest import DOCS_WITH_EMBEDDINGS


@pytest.mark.slow
@pytest.mark.generator
def test_rag_token_generator(rag_generator):
    query = "What is capital of the Germany?"
    generated_docs = rag_generator.predict(query=query, documents=DOCS_WITH_EMBEDDINGS, top_k=1)
    answers = generated_docs["answers"]
    assert len(answers) == 1
    assert "berlin" in answers[0].answer


@pytest.mark.slow
@pytest.mark.generator
@pytest.mark.parametrize("document_store", ["memory"], indirect=True)
@pytest.mark.parametrize("retriever", ["embedding"], indirect=True)
def test_generator_pipeline(document_store, retriever, rag_generator):
    document_store.write_documents(DOCS_WITH_EMBEDDINGS)
    query = "What is capital of the Germany?"
    pipeline = GenerativeQAPipeline(retriever=retriever, generator=rag_generator)
    output = pipeline.run(query=query, params={"Generator": {"top_k": 2}, "Retriever": {"top_k": 1}})
    answers = output["answers"]
    assert len(answers) == 2
    assert "berlin" in answers[0].answer


@pytest.mark.skipif(sys.platform in ['win32', 'cygwin'], reason="Gives memory allocation error on windows runner")
@pytest.mark.slow
@pytest.mark.generator
@pytest.mark.parametrize("document_store", ["memory"], indirect=True)
@pytest.mark.parametrize("retriever", ["retribert"], indirect=True)
@pytest.mark.vector_dim(128)
def test_lfqa_pipeline(document_store, retriever, eli5_generator):
    # reuse existing DOCS but regenerate embeddings with retribert
    docs: List[Document] = []
    for idx, d in enumerate(DOCS_WITH_EMBEDDINGS):
        docs.append(Document(d.content, str(idx)))
    document_store.write_documents(docs)
    document_store.update_embeddings(retriever)
    query = "Tell me about Berlin?"
    pipeline = GenerativeQAPipeline(retriever=retriever, generator=eli5_generator)
    output = pipeline.run(query=query, params={"top_k": 1})
    answers = output["answers"]
    assert len(answers) == 1
    assert "Germany" in answers[0]


@pytest.mark.slow
@pytest.mark.generator
@pytest.mark.parametrize("document_store", ["memory"], indirect=True)
@pytest.mark.parametrize("retriever", ["retribert"], indirect=True)
@pytest.mark.vector_dim(128)
def test_lfqa_pipeline_unknown_converter(document_store, retriever):
    # reuse existing DOCS but regenerate embeddings with retribert
    docs: List[Document] = []
    for idx, d in enumerate(DOCS_WITH_EMBEDDINGS):
        docs.append(Document(d.content, str(idx)))
    document_store.write_documents(docs)
    document_store.update_embeddings(retriever)
    seq2seq = Seq2SeqGenerator(model_name_or_path="patrickvonplaten/t5-tiny-random")
    query = "Tell me about Berlin?"
    pipeline = GenerativeQAPipeline(retriever=retriever, generator=seq2seq)

    # raises exception as we don't have converter for "patrickvonplaten/t5-tiny-random" in Seq2SeqGenerator
    with pytest.raises(Exception) as exception_info:
        output = pipeline.run(query=query, params={"top_k": 1})
    assert ("doesn\'t have input converter registered for patrickvonplaten/t5-tiny-random" in str(exception_info.value))


@pytest.mark.slow
@pytest.mark.generator
@pytest.mark.parametrize("document_store", ["memory"], indirect=True)
@pytest.mark.parametrize("retriever", ["retribert"], indirect=True)
@pytest.mark.vector_dim(128)
def test_lfqa_pipeline_invalid_converter(document_store, retriever):
    # reuse existing DOCS but regenerate embeddings with retribert
    docs: List[Document] = []
    for idx, d in enumerate(DOCS_WITH_EMBEDDINGS):
        docs.append(Document(d.content, str(idx)))
    document_store.write_documents(docs)
    document_store.update_embeddings(retriever)

    class _InvalidConverter:

        def __call__(self, some_invalid_para: str, another_invalid_param: str) -> None:
            pass

    seq2seq = Seq2SeqGenerator(model_name_or_path="patrickvonplaten/t5-tiny-random", input_converter=_InvalidConverter())
    query = "This query will fail due to InvalidConverter used"
    pipeline = GenerativeQAPipeline(retriever=retriever, generator=seq2seq)

    # raises exception as we are using invalid method signature in _InvalidConverter
    with pytest.raises(Exception) as exception_info:
        output = pipeline.run(query=query, params={"top_k": 1})
    assert ("does not have a valid __call__ method signature" in str(exception_info.value))


# Keeping few (retriever,document_store) combination to reduce test time
@pytest.mark.slow
@pytest.mark.generator
@pytest.mark.parametrize(
    "retriever,document_store",
    [("embedding", "memory")],
    indirect=True,
)
def test_generator_pipeline_with_translator(
    document_store,
    retriever,
    rag_generator,
    en_to_de_translator,
    de_to_en_translator
):
    document_store.write_documents(DOCS_WITH_EMBEDDINGS)
    query = "Was ist die Hauptstadt der Bundesrepublik Deutschland?"
    base_pipeline = GenerativeQAPipeline(retriever=retriever, generator=rag_generator)
    pipeline = TranslationWrapperPipeline(
        input_translator=de_to_en_translator,
        output_translator=en_to_de_translator,
        pipeline=base_pipeline
    )
    output = pipeline.run(query=query, params={"Generator": {"top_k": 2}, "Retriever": {"top_k": 1}})
    answers = output["answers"]
    assert len(answers) == 2
    assert "berlin" in answers[0].answer
