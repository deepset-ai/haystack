from haystack.document_stores import InMemoryDocumentStore
from haystack.pipelines import FAQPipeline, DocumentSearchPipeline, MostSimilarDocumentsPipeline
from haystack.nodes import EmbeddingRetriever
from haystack.schema import Document


def test_faq_pipeline_batch():
    documents = [
        {"content": "How to test module-1?", "meta": {"source": "wiki1", "answer": "Using tests for module-1"}},
        {"content": "How to test module-2?", "meta": {"source": "wiki2", "answer": "Using tests for module-2"}},
        {"content": "How to test module-3?", "meta": {"source": "wiki3", "answer": "Using tests for module-3"}},
        {"content": "How to test module-4?", "meta": {"source": "wiki4", "answer": "Using tests for module-4"}},
        {"content": "How to test module-5?", "meta": {"source": "wiki5", "answer": "Using tests for module-5"}},
    ]
    document_store = InMemoryDocumentStore()
    retriever = EmbeddingRetriever(
        document_store=document_store, embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    )
    document_store.write_documents(documents)
    document_store.update_embeddings(retriever)

    pipeline = FAQPipeline(retriever=retriever)

    output = pipeline.run_batch(queries=["How to test this?", "How to test this?"], params={"Retriever": {"top_k": 3}})
    assert len(output["answers"]) == 2  # 2 queries
    assert len(output["answers"][0]) == 3  # 3 answers per query
    assert output["queries"][0].startswith("How to")
    assert output["answers"][0][0].answer.startswith("Using tests")


def test_document_search_pipeline_batch():
    documents = [
        {"content": "Sample text for document-1", "meta": {"source": "wiki1"}},
        {"content": "Sample text for document-2", "meta": {"source": "wiki2"}},
        {"content": "Sample text for document-3", "meta": {"source": "wiki3"}},
        {"content": "Sample text for document-4", "meta": {"source": "wiki4"}},
        {"content": "Sample text for document-5", "meta": {"source": "wiki5"}},
    ]
    document_store = InMemoryDocumentStore()
    retriever = EmbeddingRetriever(
        document_store=document_store, embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    )
    document_store.write_documents(documents)
    document_store.update_embeddings(retriever)

    pipeline = DocumentSearchPipeline(retriever=retriever)
    output = pipeline.run_batch(queries=["How to test this?", "How to test this?"], params={"top_k": 4})
    assert len(output["documents"]) == 2  # 2 queries
    assert len(output["documents"][0]) == 4  # 4 docs per query


def test_most_similar_documents_pipeline_batch():
    documents = [
        {"id": "a", "content": "Sample text for document-1", "meta": {"source": "wiki1"}},
        {"id": "b", "content": "Sample text for document-2", "meta": {"source": "wiki2"}},
        {"content": "Sample text for document-3", "meta": {"source": "wiki3"}},
        {"content": "Sample text for document-4", "meta": {"source": "wiki4"}},
        {"content": "Sample text for document-5", "meta": {"source": "wiki5"}},
    ]
    document_store = InMemoryDocumentStore()
    retriever = EmbeddingRetriever(
        document_store=document_store, embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    )
    document_store.write_documents(documents)
    document_store.update_embeddings(retriever)

    docs_id: list = ["a", "b"]
    pipeline = MostSimilarDocumentsPipeline(document_store=document_store)
    list_of_documents = pipeline.run_batch(document_ids=docs_id)

    assert len(list_of_documents[0]) > 1
    assert isinstance(list_of_documents, list)
    assert len(list_of_documents) == len(docs_id)

    for another_list in list_of_documents:
        assert isinstance(another_list, list)
        for document in another_list:
            assert isinstance(document, Document)
            assert isinstance(document.id, str)
            assert isinstance(document.content, str)


def test_most_similar_documents_pipeline_with_filters_batch():
    documents = [
        {"id": "a", "content": "Sample text for document-1", "meta": {"source": "wiki1"}},
        {"id": "b", "content": "Sample text for document-2", "meta": {"source": "wiki2"}},
        {"content": "Sample text for document-3", "meta": {"source": "wiki3"}},
        {"content": "Sample text for document-4", "meta": {"source": "wiki4"}},
        {"content": "Sample text for document-5", "meta": {"source": "wiki5"}},
    ]
    document_store = InMemoryDocumentStore()
    retriever = EmbeddingRetriever(
        document_store=document_store, embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    )
    document_store = InMemoryDocumentStore()
    document_store.write_documents(documents)
    document_store.update_embeddings(retriever)

    docs_id: list = ["a", "b"]
    filters = {"source": ["wiki3", "wiki4", "wiki5"]}
    pipeline = MostSimilarDocumentsPipeline(document_store=document_store)
    list_of_documents = pipeline.run_batch(document_ids=docs_id, filters=filters)

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
