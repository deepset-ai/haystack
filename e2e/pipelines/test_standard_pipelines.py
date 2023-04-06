import os

import pytest

from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes.retriever.web import WebRetriever
from haystack.pipelines import (
    Pipeline,
    FAQPipeline,
    DocumentSearchPipeline,
    MostSimilarDocumentsPipeline,
    WebQAPipeline,
)
from haystack.nodes import EmbeddingRetriever, PromptNode
from haystack.schema import Document


def test_faq_pipeline():
    documents = [
        {"content": "How to test module-1?", "meta": {"source": "wiki1", "answer": "Using tests for module-1"}},
        {"content": "How to test module-2?", "meta": {"source": "wiki2", "answer": "Using tests for module-2"}},
        {"content": "How to test module-3?", "meta": {"source": "wiki3", "answer": "Using tests for module-3"}},
        {"content": "How to test module-4?", "meta": {"source": "wiki4", "answer": "Using tests for module-4"}},
        {"content": "How to test module-5?", "meta": {"source": "wiki5", "answer": "Using tests for module-5"}},
    ]
    document_store = InMemoryDocumentStore()
    retriever = EmbeddingRetriever(document_store=document_store, embedding_model="deepset/sentence_bert")
    document_store.write_documents(documents)
    document_store.update_embeddings(retriever)

    pipeline = FAQPipeline(retriever=retriever)

    output = pipeline.run(query="How to test this?", params={"Retriever": {"top_k": 3}})
    assert len(output["answers"]) == 3
    assert output["query"].startswith("How to")
    assert output["answers"][0].answer.startswith("Using tests")

    output = pipeline.run(
        query="How to test this?", params={"Retriever": {"filters": {"source": ["wiki2"]}, "top_k": 5}}
    )
    assert len(output["answers"]) == 1


def test_document_search_pipeline():
    documents = [
        {"content": "Sample text for document-1", "meta": {"source": "wiki1"}},
        {"content": "Sample text for document-2", "meta": {"source": "wiki2"}},
        {"content": "Sample text for document-3", "meta": {"source": "wiki3"}},
        {"content": "Sample text for document-4", "meta": {"source": "wiki4"}},
        {"content": "Sample text for document-5", "meta": {"source": "wiki5"}},
    ]
    document_store = InMemoryDocumentStore()
    retriever = EmbeddingRetriever(document_store=document_store, embedding_model="deepset/sentence_bert")
    document_store.write_documents(documents)
    document_store.update_embeddings(retriever)

    pipeline = DocumentSearchPipeline(retriever=retriever)
    output = pipeline.run(query="How to test this?", params={"top_k": 4})
    assert len(output.get("documents", [])) == 4

    output = pipeline.run(query="How to test this?", params={"filters": {"source": ["wiki2"]}, "top_k": 5})
    assert len(output["documents"]) == 1


def test_most_similar_documents_pipeline():
    documents = [
        {"id": "a", "content": "Sample text for document-1", "meta": {"source": "wiki1"}},
        {"id": "b", "content": "Sample text for document-2", "meta": {"source": "wiki2"}},
        {"content": "Sample text for document-3", "meta": {"source": "wiki3"}},
        {"content": "Sample text for document-4", "meta": {"source": "wiki4"}},
        {"content": "Sample text for document-5", "meta": {"source": "wiki5"}},
    ]
    document_store = InMemoryDocumentStore()
    retriever = EmbeddingRetriever(document_store=document_store, embedding_model="deepset/sentence_bert")
    document_store.write_documents(documents)
    document_store.update_embeddings(retriever)

    docs_id: list = ["a", "b"]
    pipeline = MostSimilarDocumentsPipeline(document_store=document_store)
    list_of_documents = pipeline.run(document_ids=docs_id)

    assert len(list_of_documents[0]) > 1
    assert isinstance(list_of_documents, list)
    assert len(list_of_documents) == len(docs_id)

    for another_list in list_of_documents:
        assert isinstance(another_list, list)
        for document in another_list:
            assert isinstance(document, Document)
            assert isinstance(document.id, str)
            assert isinstance(document.content, str)


def test_most_similar_documents_pipeline_with_filters():
    documents = [
        {"id": "a", "content": "Sample text for document-1", "meta": {"source": "wiki1"}},
        {"id": "b", "content": "Sample text for document-2", "meta": {"source": "wiki2"}},
        {"content": "Sample text for document-3", "meta": {"source": "wiki3"}},
        {"content": "Sample text for document-4", "meta": {"source": "wiki4"}},
        {"content": "Sample text for document-5", "meta": {"source": "wiki5"}},
    ]
    document_store = InMemoryDocumentStore()
    retriever = EmbeddingRetriever(document_store=document_store, embedding_model="deepset/sentence_bert")
    document_store.write_documents(documents)
    document_store.update_embeddings(retriever)

    docs_id: list = ["a", "b"]
    filters = {"source": ["wiki3", "wiki4", "wiki5"]}
    pipeline = MostSimilarDocumentsPipeline(document_store=document_store)
    list_of_documents = pipeline.run(document_ids=docs_id, filters=filters)

    assert len(list_of_documents[0]) > 1
    assert isinstance(list_of_documents, list)
    assert len(list_of_documents) == len(docs_id)

    for another_list in list_of_documents:
        assert isinstance(another_list, list)
        for document in another_list:
            assert isinstance(document, Document)
            assert isinstance(document.id, str)
            assert isinstance(document.content, str)
            assert document.meta["source"] in ["wiki3", "wiki4", "wiki5"]


def test_query_and_indexing_pipeline(samples_path):
    # test correct load of indexing pipeline from yaml
    pipeline = Pipeline.load_from_yaml(
        samples_path / "pipelines" / "test.haystack-pipeline.yml", pipeline_name="indexing_pipeline"
    )
    pipeline.run(file_paths=samples_path / "pipelines" / "sample_pdf_1.pdf")
    # test correct load of query pipeline from yaml
    pipeline = Pipeline.load_from_yaml(
        samples_path / "pipelines" / "test.haystack-pipeline.yml", pipeline_name="query_pipeline"
    )
    prediction = pipeline.run(
        query="Who made the PDF specification?", params={"Retriever": {"top_k": 2}, "Reader": {"top_k": 1}}
    )
    assert prediction["query"] == "Who made the PDF specification?"
    assert prediction["answers"][0].answer == "Adobe Systems"
    assert prediction["answers"][0].meta["classification"]["label"] == "joy"
    assert "_debug" not in prediction.keys()


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY", None),
    reason="Please export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
)
@pytest.mark.skipif(
    not os.environ.get("SERPERDEV_API_KEY", None),
    reason="Please export an env var called SERPERDEV_API_KEY containing the SerperDev key to run this test.",
)
def test_webqa_pipeline():
    search_key = os.environ.get("SERPERDEV_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    pn = PromptNode(
        "text-davinci-003",
        api_key=openai_key,
        max_length=256,
        default_prompt_template="question-answering-with-document-scores",
    )
    web_retriever = WebRetriever(api_key=search_key, top_search_results=2)
    pipeline = WebQAPipeline(retriever=web_retriever, prompt_node=pn)
    result = pipeline.run(query="Who is the father of Arya Stark?")
    assert isinstance(result, dict)
    assert len(result["results"]) == 1
    answer = result["results"][0]
    assert "Stark" in answer or "NED" in answer
