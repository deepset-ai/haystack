import os
import sys
from typing import List

import pytest

from haystack.schema import Document
from haystack.nodes.answer_generator import Seq2SeqGenerator, OpenAIAnswerGenerator
from haystack.pipelines import TranslationWrapperPipeline, GenerativeQAPipeline
from haystack.nodes import PromptTemplate

import logging


# Keeping few (retriever,document_store) combination to reduce test time
@pytest.mark.skipif(sys.platform in ["win32", "cygwin"], reason="Causes OOM on windows github runner")
@pytest.mark.integration
@pytest.mark.generator
@pytest.mark.parametrize("retriever,document_store", [("embedding", "memory")], indirect=True)
def test_generator_pipeline_with_translator(
    document_store, retriever, rag_generator, en_to_de_translator, de_to_en_translator, docs_with_true_emb
):
    document_store.write_documents(docs_with_true_emb)
    query = "Was ist die Hauptstadt der Bundesrepublik Deutschland?"
    base_pipeline = GenerativeQAPipeline(retriever=retriever, generator=rag_generator)
    pipeline = TranslationWrapperPipeline(
        input_translator=de_to_en_translator, output_translator=en_to_de_translator, pipeline=base_pipeline
    )
    output = pipeline.run(query=query, params={"Generator": {"top_k": 2}, "Retriever": {"top_k": 1}})
    answers = output["answers"]
    assert len(answers) == 2
    assert "berlin" in answers[0].answer


@pytest.mark.integration
@pytest.mark.generator
def test_rag_token_generator(rag_generator, docs_with_true_emb):
    query = "What is capital of the Germany?"
    generated_docs = rag_generator.predict(query=query, documents=docs_with_true_emb, top_k=1)
    answers = generated_docs["answers"]
    assert len(answers) == 1
    assert "berlin" in answers[0].answer


@pytest.mark.integration
@pytest.mark.generator
@pytest.mark.parametrize("document_store", ["memory"], indirect=True)
@pytest.mark.parametrize("retriever", ["embedding"], indirect=True)
def test_generator_pipeline(document_store, retriever, rag_generator, docs_with_true_emb):
    document_store.write_documents(docs_with_true_emb)
    query = "What is capital of the Germany?"
    pipeline = GenerativeQAPipeline(retriever=retriever, generator=rag_generator)
    output = pipeline.run(query=query, params={"Generator": {"top_k": 2}, "Retriever": {"top_k": 1}})
    answers = output["answers"]
    assert len(answers) == 2
    assert "berlin" in answers[0].answer
    for doc_idx, document in enumerate(output["documents"]):
        assert document.id == answers[0].document_ids[doc_idx]
        assert document.meta == answers[0].meta["doc_metas"][doc_idx]


@pytest.mark.skipif(sys.platform in ["win32", "cygwin"], reason="Causes OOM on windows github runner")
@pytest.mark.integration
@pytest.mark.generator
@pytest.mark.parametrize("document_store", ["memory"], indirect=True)
@pytest.mark.parametrize("retriever", ["retribert", "dpr_lfqa"], indirect=True)
@pytest.mark.parametrize("lfqa_generator", ["yjernite/bart_eli5", "vblagoje/bart_lfqa"], indirect=True)
@pytest.mark.embedding_dim(128)
def test_lfqa_pipeline(document_store, retriever, lfqa_generator, docs_with_true_emb):
    # reuse existing DOCS but regenerate embeddings with retribert
    docs: List[Document] = []
    for d in docs_with_true_emb:
        docs.append(Document(content=d.content))
    document_store.write_documents(docs)
    document_store.update_embeddings(retriever)
    query = "Tell me about Berlin?"
    pipeline = GenerativeQAPipeline(generator=lfqa_generator, retriever=retriever)
    output = pipeline.run(query=query, params={"top_k": 1})
    answers = output["answers"]
    assert len(answers) == 1, answers
    assert "Germany" in answers[0].answer, answers[0].answer


@pytest.mark.integration
@pytest.mark.generator
@pytest.mark.parametrize("document_store", ["memory"], indirect=True)
@pytest.mark.parametrize("retriever", ["retribert"], indirect=True)
@pytest.mark.embedding_dim(128)
def test_lfqa_pipeline_unknown_converter(document_store, retriever, docs_with_true_emb):
    # reuse existing DOCS but regenerate embeddings with retribert
    docs: List[Document] = []
    for d in docs_with_true_emb:
        docs.append(Document(content=d.content))
    document_store.write_documents(docs)
    document_store.update_embeddings(retriever)
    seq2seq = Seq2SeqGenerator(model_name_or_path="patrickvonplaten/t5-tiny-random")
    query = "Tell me about Berlin?"
    pipeline = GenerativeQAPipeline(retriever=retriever, generator=seq2seq)

    # raises exception as we don't have converter for "patrickvonplaten/t5-tiny-random" in Seq2SeqGenerator
    with pytest.raises(Exception) as exception_info:
        output = pipeline.run(query=query, params={"top_k": 1})
    assert "doesn't have input converter registered for patrickvonplaten/t5-tiny-random" in str(exception_info.value)


@pytest.mark.integration
@pytest.mark.generator
@pytest.mark.parametrize("document_store", ["memory"], indirect=True)
@pytest.mark.parametrize("retriever", ["retribert"], indirect=True)
@pytest.mark.embedding_dim(128)
def test_lfqa_pipeline_invalid_converter(document_store, retriever, docs_with_true_emb):
    # reuse existing DOCS but regenerate embeddings with retribert
    docs: List[Document] = []
    for d in docs_with_true_emb:
        docs.append(Document(content=d.content))
    document_store.write_documents(docs)
    document_store.update_embeddings(retriever)

    class _InvalidConverter:
        def __call__(self, some_invalid_para: str, another_invalid_param: str) -> None:
            pass

    seq2seq = Seq2SeqGenerator(
        model_name_or_path="patrickvonplaten/t5-tiny-random", input_converter=_InvalidConverter()
    )
    query = "This query will fail due to InvalidConverter used"
    pipeline = GenerativeQAPipeline(retriever=retriever, generator=seq2seq)

    # raises exception as we are using invalid method signature in _InvalidConverter
    with pytest.raises(Exception) as exception_info:
        output = pipeline.run(query=query, params={"top_k": 1})
    assert "does not have a valid __call__ method signature" in str(exception_info.value)


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY", None),
    reason="No OpenAI API key provided. Please export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
)
def test_openai_answer_generator(openai_generator, docs):
    prediction = openai_generator.predict(query="Who lives in Berlin?", documents=docs, top_k=1)
    assert len(prediction["answers"]) == 1
    assert "Carla" in prediction["answers"][0].answer


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY", None),
    reason="No OpenAI API key provided. Please export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
)
def test_openai_answer_generator_custom_template(docs):
    lfqa_prompt = PromptTemplate(
        name="lfqa",
        prompt_text="""
        Synthesize a comprehensive answer from your knowledge and the following topk most relevant paragraphs and the given question.
        \n===\Paragraphs: $context\n===\n$query""",
        prompt_params=["context", "query"],
    )
    node = OpenAIAnswerGenerator(
        api_key=os.environ.get("OPENAI_API_KEY", ""), model="text-babbage-001", top_k=1, prompt_template=lfqa_prompt
    )
    prediction = node.predict(query="Who lives in Berlin?", documents=docs, top_k=1)
    assert len(prediction["answers"]) == 1


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY", None),
    reason="No OpenAI API key provided. Please export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
)
def test_openai_answer_generator_max_token(docs, caplog):
    openai_generator = OpenAIAnswerGenerator(
        api_key=os.environ.get("OPENAI_API_KEY", ""), model="text-babbage-001", top_k=1
    )
    openai_generator.MAX_TOKENS_LIMIT = 116
    with caplog.at_level(logging.INFO):
        prediction = openai_generator.predict(query="Who lives in Berlin?", documents=docs, top_k=1)
        assert "Skipping all of the provided Documents" in caplog.text
        assert len(prediction["answers"]) == 1
        # Can't easily check content of answer since it is generative and can change between runs
