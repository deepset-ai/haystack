import pytest

from haystack.schema import Document
from haystack.document_stores import InMemoryDocumentStore
from haystack.pipelines import GenerativeQAPipeline, TranslationWrapperPipeline
from haystack.nodes import BM25Retriever, RAGenerator, DensePassageRetriever, TransformersTranslator


@pytest.fixture
def docs():
    return [
        Document(content="The capital of Germany is the city state of Berlin."),
        Document(content="Berlin is the capital and largest city of Germany by both area and population."),
    ]


def test_rag_generator_pipeline(docs):
    document_store = InMemoryDocumentStore(use_bm25=True)
    retriever = DensePassageRetriever(
        document_store=document_store,
        query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
        passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
        embed_title=True,
    )
    rag_generator = RAGenerator(model_name_or_path="facebook/rag-token-nq", generator_type="token", max_length=20)
    document_store.write_documents(docs)

    query = "What is capital of the Germany?"
    pipeline = GenerativeQAPipeline(retriever=retriever, generator=rag_generator)
    output = pipeline.run(query=query, params={"Generator": {"top_k": 2}, "Retriever": {"top_k": 1}})
    answers = output["answers"]
    assert len(answers) == 2
    assert "berlin" in answers[0].answer.lower()


def test_rag_generator_pipeline_with_translator():
    docs = [
        Document(content="The capital of Germany is the city state of Berlin."),
        Document(content="Berlin is the capital and largest city of Germany by both area and population."),
    ]
    ds = InMemoryDocumentStore(use_bm25=True)
    ds.write_documents(docs)
    retriever = DensePassageRetriever(  # Needs DPR or RAGenerator will thrown an exception...
        document_store=ds,
        query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
        passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
        use_gpu=False,
        embed_title=True,
    )
    ds.update_embeddings(retriever=retriever)
    rag_generator = RAGenerator(
        model_name_or_path="facebook/rag-token-nq", generator_type="token", max_length=20, retriever=retriever
    )
    en_to_de_translator = TransformersTranslator(model_name_or_path="Helsinki-NLP/opus-mt-en-de")
    de_to_en_translator = TransformersTranslator(model_name_or_path="Helsinki-NLP/opus-mt-de-en")

    query = "Was ist die Hauptstadt der Bundesrepublik Deutschland?"
    base_pipeline = GenerativeQAPipeline(retriever=retriever, generator=rag_generator)
    pipeline = TranslationWrapperPipeline(
        input_translator=de_to_en_translator, output_translator=en_to_de_translator, pipeline=base_pipeline
    )
    output = pipeline.run(query=query, params={"Generator": {"top_k": 2}, "Retriever": {"top_k": 1}})
    answers = output["answers"]
    assert len(answers) == 2
    assert "berlin" in answers[0].answer.lower()
    for doc_idx, document in enumerate(output["documents"]):
        assert document.id == answers[0].document_ids[doc_idx]
        assert document.meta == answers[0].meta["doc_metas"][doc_idx]
