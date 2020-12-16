import pytest

from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.pipeline import JoinDocuments, ExtractiveQAPipeline, Pipeline, FAQPipeline, DocumentSearchPipeline
from haystack.retriever.dense import DensePassageRetriever
from haystack.retriever.sparse import ElasticsearchRetriever


@pytest.mark.slow
@pytest.mark.elasticsearch
@pytest.mark.parametrize("retriever_with_docs", ["elasticsearch"], indirect=True)
def test_graph_creation(reader, retriever_with_docs, document_store_with_docs):
    pipeline = Pipeline()
    pipeline.add_node(name="ES", component=retriever_with_docs, inputs=["Query"])

    with pytest.raises(AssertionError):
        pipeline.add_node(name="Reader", component=retriever_with_docs, inputs=["ES.output_2"])

    with pytest.raises(AssertionError):
        pipeline.add_node(name="Reader", component=retriever_with_docs, inputs=["ES.wrong_edge_label"])

    with pytest.raises(Exception):
        pipeline.add_node(name="Reader", component=retriever_with_docs, inputs=["InvalidNode"])


@pytest.mark.slow
@pytest.mark.elasticsearch
@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
def test_extractive_qa_answers(reader, retriever_with_docs):
    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever_with_docs)
    prediction = pipeline.run(query="Who lives in Berlin?", top_k_retriever=10, top_k_reader=3)
    assert prediction is not None
    assert prediction["query"] == "Who lives in Berlin?"
    assert prediction["answers"][0]["answer"] == "Carla"
    assert prediction["answers"][0]["probability"] <= 1
    assert prediction["answers"][0]["probability"] >= 0
    assert prediction["answers"][0]["meta"]["meta_field"] == "test1"
    assert prediction["answers"][0]["context"] == "My name is Carla and I live in Berlin"

    assert len(prediction["answers"]) == 3


@pytest.mark.elasticsearch
@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
def test_extractive_qa_offsets(reader, retriever_with_docs):
    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever_with_docs)
    prediction = pipeline.run(query="Who lives in Berlin?", top_k_retriever=10, top_k_reader=5)

    assert prediction["answers"][0]["offset_start"] == 11
    assert prediction["answers"][0]["offset_end"] == 16
    start = prediction["answers"][0]["offset_start"]
    end = prediction["answers"][0]["offset_end"]
    assert prediction["answers"][0]["context"][start:end] == prediction["answers"][0]["answer"]


@pytest.mark.slow
@pytest.mark.elasticsearch
@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
def test_extractive_qa_answers_single_result(reader, retriever_with_docs):
    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever_with_docs)
    query = "testing finder"
    prediction = pipeline.run(query=query, top_k_retriever=1, top_k_reader=1)
    assert prediction is not None
    assert len(prediction["answers"]) == 1


@pytest.mark.elasticsearch
@pytest.mark.parametrize(
    "retriever,document_store",
    [("embedding", "memory"), ("embedding", "faiss"), ("embedding", "elasticsearch")],
    indirect=True,
)
def test_faq_pipeline(retriever, document_store):
    documents = [
        {"text": "How to test module-1?", 'meta': {"source": "wiki1", "answer": "Using tests for module-1"}},
        {"text": "How to test module-2?", 'meta': {"source": "wiki2", "answer": "Using tests for module-2"}},
        {"text": "How to test module-3?", 'meta': {"source": "wiki3", "answer": "Using tests for module-3"}},
        {"text": "How to test module-4?", 'meta': {"source": "wiki4", "answer": "Using tests for module-4"}},
        {"text": "How to test module-5?", 'meta': {"source": "wiki5", "answer": "Using tests for module-5"}},
    ]

    document_store.write_documents(documents)
    document_store.update_embeddings(retriever)

    pipeline = FAQPipeline(retriever=retriever)

    output = pipeline.run(query="How to test this?", top_k_retriever=3)
    assert len(output["answers"]) == 3
    assert output["answers"][0]["query"].startswith("How to")
    assert output["answers"][0]["answer"].startswith("Using tests")

    if isinstance(document_store, ElasticsearchDocumentStore):
        output = pipeline.run(query="How to test this?", filters={"source": ["wiki2"]}, top_k_retriever=5)
        assert len(output["answers"]) == 1


@pytest.mark.elasticsearch
@pytest.mark.parametrize(
    "retriever,document_store",
    [("embedding", "memory"), ("embedding", "faiss"), ("embedding", "elasticsearch")],
    indirect=True,
)
def test_document_search_pipeline(retriever, document_store):
    documents = [
        {"text": "Sample text for document-1", 'meta': {"source": "wiki1"}},
        {"text": "Sample text for document-2", 'meta': {"source": "wiki2"}},
        {"text": "Sample text for document-3", 'meta': {"source": "wiki3"}},
        {"text": "Sample text for document-4", 'meta': {"source": "wiki4"}},
        {"text": "Sample text for document-5", 'meta': {"source": "wiki5"}},
    ]

    document_store.write_documents(documents)
    document_store.update_embeddings(retriever)

    pipeline = DocumentSearchPipeline(retriever=retriever)
    output = pipeline.run(query="How to test this?", top_k_retriever=4)
    assert len(output.get('documents', [])) == 4

    if isinstance(document_store, ElasticsearchDocumentStore):
        output = pipeline.run(query="How to test this?", filters={"source": ["wiki2"]}, top_k_retriever=5)
        assert len(output["documents"]) == 1


@pytest.mark.parametrize("document_store_with_docs", ["elasticsearch"], indirect=True)
@pytest.mark.parametrize("reader", ["farm"], indirect=True)
def test_join_document_pipeline(document_store_with_docs, reader):
    es = ElasticsearchRetriever(document_store=document_store_with_docs)
    dpr = DensePassageRetriever(
        document_store=document_store_with_docs,
        query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
        passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
        use_gpu=False,
    )
    document_store_with_docs.update_embeddings(dpr)

    query = "Where does Carla lives?"

    # test merge without weights
    join_node = JoinDocuments(join_mode="merge")
    p = Pipeline()
    p.add_node(component=es, name="R1", inputs=["Query"])
    p.add_node(component=dpr, name="R2", inputs=["Query"])
    p.add_node(component=join_node, name="Join", inputs=["R1", "R2"])
    results = p.run(query=query)
    assert len(results["documents"]) == 3

    # test merge with weights
    join_node = JoinDocuments(join_mode="merge", weights=[1000, 1], top_k_join=2)
    p = Pipeline()
    p.add_node(component=es, name="R1", inputs=["Query"])
    p.add_node(component=dpr, name="R2", inputs=["Query"])
    p.add_node(component=join_node, name="Join", inputs=["R1", "R2"])
    results = p.run(query=query)
    assert results["documents"][0].score > 1000
    assert len(results["documents"]) == 2

    # test concatenate
    join_node = JoinDocuments(join_mode="concatenate")
    p = Pipeline()
    p.add_node(component=es, name="R1", inputs=["Query"])
    p.add_node(component=dpr, name="R2", inputs=["Query"])
    p.add_node(component=join_node, name="Join", inputs=["R1", "R2"])
    results = p.run(query=query)
    assert len(results["documents"]) == 3

    # test join_node with reader
    join_node = JoinDocuments()
    p = Pipeline()
    p.add_node(component=es, name="R1", inputs=["Query"])
    p.add_node(component=dpr, name="R2", inputs=["Query"])
    p.add_node(component=join_node, name="Join", inputs=["R1", "R2"])
    p.add_node(component=reader, name="Reader", inputs=["Join"])
    results = p.run(query=query)
    assert results["answers"][0]["answer"] == "Berlin"
