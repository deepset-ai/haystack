from pathlib import Path
from collections import defaultdict
from unittest.mock import Mock

import os
import math
import pytest

from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack.pipelines import Pipeline, FAQPipeline, DocumentSearchPipeline, RootNode, MostSimilarDocumentsPipeline
from haystack.nodes import (
    DensePassageRetriever,
    BM25Retriever,
    SklearnQueryClassifier,
    TransformersQueryClassifier,
    EmbeddingRetriever,
    JoinDocuments,
)
from haystack.schema import Document

from ..conftest import SAMPLES_PATH


@pytest.mark.parametrize(
    "retriever,document_store",
    [("embedding", "memory"), ("embedding", "faiss"), ("embedding", "milvus1"), ("embedding", "elasticsearch")],
    indirect=True,
)
def test_faq_pipeline(retriever, document_store):
    documents = [
        {"content": "How to test module-1?", "meta": {"source": "wiki1", "answer": "Using tests for module-1"}},
        {"content": "How to test module-2?", "meta": {"source": "wiki2", "answer": "Using tests for module-2"}},
        {"content": "How to test module-3?", "meta": {"source": "wiki3", "answer": "Using tests for module-3"}},
        {"content": "How to test module-4?", "meta": {"source": "wiki4", "answer": "Using tests for module-4"}},
        {"content": "How to test module-5?", "meta": {"source": "wiki5", "answer": "Using tests for module-5"}},
    ]

    document_store.write_documents(documents)
    document_store.update_embeddings(retriever)

    pipeline = FAQPipeline(retriever=retriever)

    output = pipeline.run(query="How to test this?", params={"Retriever": {"top_k": 3}})
    assert len(output["answers"]) == 3
    assert output["query"].startswith("How to")
    assert output["answers"][0].answer.startswith("Using tests")

    if isinstance(document_store, ElasticsearchDocumentStore):
        output = pipeline.run(
            query="How to test this?", params={"Retriever": {"filters": {"source": ["wiki2"]}, "top_k": 5}}
        )
        assert len(output["answers"]) == 1


@pytest.mark.parametrize("retriever,document_store", [("embedding", "memory")], indirect=True)
def test_faq_pipeline_batch(retriever, document_store):
    documents = [
        {"content": "How to test module-1?", "meta": {"source": "wiki1", "answer": "Using tests for module-1"}},
        {"content": "How to test module-2?", "meta": {"source": "wiki2", "answer": "Using tests for module-2"}},
        {"content": "How to test module-3?", "meta": {"source": "wiki3", "answer": "Using tests for module-3"}},
        {"content": "How to test module-4?", "meta": {"source": "wiki4", "answer": "Using tests for module-4"}},
        {"content": "How to test module-5?", "meta": {"source": "wiki5", "answer": "Using tests for module-5"}},
    ]

    document_store.write_documents(documents)
    document_store.update_embeddings(retriever)

    pipeline = FAQPipeline(retriever=retriever)

    output = pipeline.run_batch(queries=["How to test this?", "How to test this?"], params={"Retriever": {"top_k": 3}})
    assert len(output["answers"]) == 2  # 2 queries
    assert len(output["answers"][0]) == 3  # 3 answers per query
    assert output["queries"][0].startswith("How to")
    assert output["answers"][0][0].answer.startswith("Using tests")


@pytest.mark.parametrize("retriever", ["embedding"], indirect=True)
@pytest.mark.parametrize(
    "document_store", ["elasticsearch", "faiss", "memory", "milvus1", "milvus", "weaviate", "pinecone"], indirect=True
)
def test_document_search_pipeline(retriever, document_store):
    documents = [
        {"content": "Sample text for document-1", "meta": {"source": "wiki1"}},
        {"content": "Sample text for document-2", "meta": {"source": "wiki2"}},
        {"content": "Sample text for document-3", "meta": {"source": "wiki3"}},
        {"content": "Sample text for document-4", "meta": {"source": "wiki4"}},
        {"content": "Sample text for document-5", "meta": {"source": "wiki5"}},
    ]

    document_store.write_documents(documents)
    document_store.update_embeddings(retriever)

    pipeline = DocumentSearchPipeline(retriever=retriever)
    output = pipeline.run(query="How to test this?", params={"top_k": 4})
    assert len(output.get("documents", [])) == 4

    if isinstance(document_store, ElasticsearchDocumentStore):
        output = pipeline.run(query="How to test this?", params={"filters": {"source": ["wiki2"]}, "top_k": 5})
        assert len(output["documents"]) == 1


@pytest.mark.parametrize("retriever", ["embedding"], indirect=True)
@pytest.mark.parametrize("document_store", ["memory"], indirect=True)
def test_document_search_pipeline_batch(retriever, document_store):
    documents = [
        {"content": "Sample text for document-1", "meta": {"source": "wiki1"}},
        {"content": "Sample text for document-2", "meta": {"source": "wiki2"}},
        {"content": "Sample text for document-3", "meta": {"source": "wiki3"}},
        {"content": "Sample text for document-4", "meta": {"source": "wiki4"}},
        {"content": "Sample text for document-5", "meta": {"source": "wiki5"}},
    ]

    document_store.write_documents(documents)
    document_store.update_embeddings(retriever)

    pipeline = DocumentSearchPipeline(retriever=retriever)
    output = pipeline.run_batch(queries=["How to test this?", "How to test this?"], params={"top_k": 4})
    assert len(output["documents"]) == 2  # 2 queries
    assert len(output["documents"][0]) == 4  # 4 docs per query


@pytest.mark.integration
@pytest.mark.parametrize("retriever_with_docs", ["elasticsearch", "dpr", "embedding"], indirect=True)
@pytest.mark.parametrize("document_store_with_docs", ["elasticsearch"], indirect=True)
def test_documentsearch_es_authentication(retriever_with_docs, document_store_with_docs: ElasticsearchDocumentStore):
    if isinstance(retriever_with_docs, (DensePassageRetriever, EmbeddingRetriever)):
        document_store_with_docs.update_embeddings(retriever=retriever_with_docs)
    mock_client = Mock(wraps=document_store_with_docs.client)
    document_store_with_docs.client = mock_client
    auth_headers = {"Authorization": "Basic YWRtaW46cm9vdA=="}
    pipeline = DocumentSearchPipeline(retriever=retriever_with_docs)
    prediction = pipeline.run(
        query="Who lives in Berlin?", params={"Retriever": {"top_k": 10, "headers": auth_headers}}
    )
    assert prediction is not None
    assert len(prediction["documents"]) == 5
    mock_client.search.assert_called_once()
    args, kwargs = mock_client.search.call_args
    assert "headers" in kwargs
    assert kwargs["headers"] == auth_headers


@pytest.mark.integration
@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
def test_documentsearch_document_store_authentication(retriever_with_docs, document_store_with_docs):
    mock_client = None
    if isinstance(document_store_with_docs, ElasticsearchDocumentStore):
        es_document_store: ElasticsearchDocumentStore = document_store_with_docs
        mock_client = Mock(wraps=es_document_store.client)
        es_document_store.client = mock_client
    auth_headers = {"Authorization": "Basic YWRtaW46cm9vdA=="}
    pipeline = DocumentSearchPipeline(retriever=retriever_with_docs)
    if not mock_client:
        with pytest.raises(Exception):
            prediction = pipeline.run(
                query="Who lives in Berlin?", params={"Retriever": {"top_k": 10, "headers": auth_headers}}
            )
    else:
        prediction = pipeline.run(
            query="Who lives in Berlin?", params={"Retriever": {"top_k": 10, "headers": auth_headers}}
        )
        assert prediction is not None
        assert len(prediction["documents"]) == 5
        mock_client.count.assert_called_once()
        args, kwargs = mock_client.count.call_args
        assert "headers" in kwargs
        assert kwargs["headers"] == auth_headers


@pytest.mark.parametrize(
    "retriever,document_store",
    [("embedding", "faiss"), ("embedding", "milvus1"), ("embedding", "elasticsearch")],
    indirect=True,
)
def test_most_similar_documents_pipeline(retriever, document_store):
    documents = [
        {"id": "a", "content": "Sample text for document-1", "meta": {"source": "wiki1"}},
        {"id": "b", "content": "Sample text for document-2", "meta": {"source": "wiki2"}},
        {"content": "Sample text for document-3", "meta": {"source": "wiki3"}},
        {"content": "Sample text for document-4", "meta": {"source": "wiki4"}},
        {"content": "Sample text for document-5", "meta": {"source": "wiki5"}},
    ]

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


@pytest.mark.parametrize(
    "retriever,document_store", [("embedding", "milvus1"), ("embedding", "elasticsearch")], indirect=True
)
def test_most_similar_documents_pipeline_with_filters(retriever, document_store):
    documents = [
        {"id": "a", "content": "Sample text for document-1", "meta": {"source": "wiki1"}},
        {"id": "b", "content": "Sample text for document-2", "meta": {"source": "wiki2"}},
        {"content": "Sample text for document-3", "meta": {"source": "wiki3"}},
        {"content": "Sample text for document-4", "meta": {"source": "wiki4"}},
        {"content": "Sample text for document-5", "meta": {"source": "wiki5"}},
    ]

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


@pytest.mark.parametrize("retriever,document_store", [("embedding", "memory")], indirect=True)
def test_most_similar_documents_pipeline_batch(retriever, document_store):
    documents = [
        {"id": "a", "content": "Sample text for document-1", "meta": {"source": "wiki1"}},
        {"id": "b", "content": "Sample text for document-2", "meta": {"source": "wiki2"}},
        {"content": "Sample text for document-3", "meta": {"source": "wiki3"}},
        {"content": "Sample text for document-4", "meta": {"source": "wiki4"}},
        {"content": "Sample text for document-5", "meta": {"source": "wiki5"}},
    ]

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


@pytest.mark.parametrize("retriever,document_store", [("embedding", "memory")], indirect=True)
def test_most_similar_documents_pipeline_with_filters_batch(retriever, document_store):
    documents = [
        {"id": "a", "content": "Sample text for document-1", "meta": {"source": "wiki1"}},
        {"id": "b", "content": "Sample text for document-2", "meta": {"source": "wiki2"}},
        {"content": "Sample text for document-3", "meta": {"source": "wiki3"}},
        {"content": "Sample text for document-4", "meta": {"source": "wiki4"}},
        {"content": "Sample text for document-5", "meta": {"source": "wiki5"}},
    ]

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


@pytest.mark.integration
@pytest.mark.parametrize("document_store_with_docs", ["memory"], indirect=True)
def test_most_similar_documents_pipeline_save(tmpdir, document_store_with_docs):
    pipeline = MostSimilarDocumentsPipeline(document_store=document_store_with_docs)
    path = Path(tmpdir, "most_similar_document_pipeline.yml")
    pipeline.save_to_yaml(path)
    os.path.exists(path)


@pytest.mark.elasticsearch
@pytest.mark.parametrize("document_store_dot_product_with_docs", ["elasticsearch"], indirect=True)
def test_join_merge_no_weights(document_store_dot_product_with_docs):
    es = BM25Retriever(document_store=document_store_dot_product_with_docs)
    dpr = DensePassageRetriever(
        document_store=document_store_dot_product_with_docs,
        query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
        passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
        use_gpu=False,
    )
    document_store_dot_product_with_docs.update_embeddings(dpr)

    query = "Where does Carla live?"

    join_node = JoinDocuments(join_mode="merge")
    p = Pipeline()
    p.add_node(component=es, name="R1", inputs=["Query"])
    p.add_node(component=dpr, name="R2", inputs=["Query"])
    p.add_node(component=join_node, name="Join", inputs=["R1", "R2"])
    results = p.run(query=query)
    assert len(results["documents"]) == 5


@pytest.mark.elasticsearch
@pytest.mark.parametrize("document_store_dot_product_with_docs", ["elasticsearch"], indirect=True)
def test_join_merge_with_weights(document_store_dot_product_with_docs):
    es = BM25Retriever(document_store=document_store_dot_product_with_docs)
    dpr = DensePassageRetriever(
        document_store=document_store_dot_product_with_docs,
        query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
        passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
        use_gpu=False,
    )
    document_store_dot_product_with_docs.update_embeddings(dpr)

    query = "Where does Carla live?"

    join_node = JoinDocuments(join_mode="merge", weights=[1000, 1], top_k_join=2)
    p = Pipeline()
    p.add_node(component=es, name="R1", inputs=["Query"])
    p.add_node(component=dpr, name="R2", inputs=["Query"])
    p.add_node(component=join_node, name="Join", inputs=["R1", "R2"])
    results = p.run(query=query)
    assert math.isclose(results["documents"][0].score, 0.5481393431183286, rel_tol=0.0001)
    assert len(results["documents"]) == 2


@pytest.mark.elasticsearch
@pytest.mark.parametrize("document_store_dot_product_with_docs", ["elasticsearch"], indirect=True)
def test_join_concatenate(document_store_dot_product_with_docs):
    es = BM25Retriever(document_store=document_store_dot_product_with_docs)
    dpr = DensePassageRetriever(
        document_store=document_store_dot_product_with_docs,
        query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
        passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
        use_gpu=False,
    )
    document_store_dot_product_with_docs.update_embeddings(dpr)

    query = "Where does Carla live?"

    join_node = JoinDocuments(join_mode="concatenate")
    p = Pipeline()
    p.add_node(component=es, name="R1", inputs=["Query"])
    p.add_node(component=dpr, name="R2", inputs=["Query"])
    p.add_node(component=join_node, name="Join", inputs=["R1", "R2"])
    results = p.run(query=query)
    assert len(results["documents"]) == 5


@pytest.mark.elasticsearch
@pytest.mark.parametrize("document_store_dot_product_with_docs", ["elasticsearch"], indirect=True)
def test_join_concatenate_with_topk(document_store_dot_product_with_docs):
    es = BM25Retriever(document_store=document_store_dot_product_with_docs)
    dpr = DensePassageRetriever(
        document_store=document_store_dot_product_with_docs,
        query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
        passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
        use_gpu=False,
    )
    document_store_dot_product_with_docs.update_embeddings(dpr)

    query = "Where does Carla live?"

    join_node = JoinDocuments(join_mode="concatenate")
    p = Pipeline()
    p.add_node(component=es, name="R1", inputs=["Query"])
    p.add_node(component=dpr, name="R2", inputs=["Query"])
    p.add_node(component=join_node, name="Join", inputs=["R1", "R2"])
    one_result = p.run(query=query, params={"Join": {"top_k_join": 1}})
    two_results = p.run(query=query, params={"Join": {"top_k_join": 2}})
    assert len(one_result["documents"]) == 1
    assert len(two_results["documents"]) == 2


@pytest.mark.elasticsearch
@pytest.mark.parametrize("document_store_dot_product_with_docs", ["elasticsearch"], indirect=True)
@pytest.mark.parametrize("reader", ["farm"], indirect=True)
def test_join_with_reader(document_store_dot_product_with_docs, reader):
    es = BM25Retriever(document_store=document_store_dot_product_with_docs)
    dpr = DensePassageRetriever(
        document_store=document_store_dot_product_with_docs,
        query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
        passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
        use_gpu=False,
    )
    document_store_dot_product_with_docs.update_embeddings(dpr)

    query = "Where does Carla live?"

    join_node = JoinDocuments()
    p = Pipeline()
    p.add_node(component=es, name="R1", inputs=["Query"])
    p.add_node(component=dpr, name="R2", inputs=["Query"])
    p.add_node(component=join_node, name="Join", inputs=["R1", "R2"])
    p.add_node(component=reader, name="Reader", inputs=["Join"])
    results = p.run(query=query)
    # check whether correct answer is within top 2 predictions
    assert results["answers"][0].answer == "Berlin" or results["answers"][1].answer == "Berlin"


@pytest.mark.elasticsearch
@pytest.mark.parametrize("document_store_dot_product_with_docs", ["elasticsearch"], indirect=True)
def test_join_with_rrf(document_store_dot_product_with_docs):
    es = BM25Retriever(document_store=document_store_dot_product_with_docs)
    dpr = DensePassageRetriever(
        document_store=document_store_dot_product_with_docs,
        query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
        passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
        use_gpu=False,
    )
    document_store_dot_product_with_docs.update_embeddings(dpr)

    query = "Where does Carla live?"

    join_node = JoinDocuments(join_mode="reciprocal_rank_fusion")
    p = Pipeline()
    p.add_node(component=es, name="R1", inputs=["Query"])
    p.add_node(component=dpr, name="R2", inputs=["Query"])
    p.add_node(component=join_node, name="Join", inputs=["R1", "R2"])
    results = p.run(query=query)

    # list of precalculated expected results
    expected_scores = [
        0.03278688524590164,
        0.03200204813108039,
        0.03200204813108039,
        0.031009615384615385,
        0.031009615384615385,
    ]

    assert all([doc.score == expected_scores[idx] for idx, doc in enumerate(results["documents"])])


def test_query_keyword_statement_classifier():
    class KeywordOutput(RootNode):
        outgoing_edges = 2

        def run(self, **kwargs):
            kwargs["output"] = "keyword"
            return kwargs, "output_1"

    class QuestionOutput(RootNode):
        outgoing_edges = 2

        def run(self, **kwargs):
            kwargs["output"] = "question"
            return kwargs, "output_2"

    pipeline = Pipeline()
    pipeline.add_node(name="SkQueryKeywordQuestionClassifier", component=SklearnQueryClassifier(), inputs=["Query"])
    pipeline.add_node(
        name="KeywordNode", component=KeywordOutput(), inputs=["SkQueryKeywordQuestionClassifier.output_2"]
    )
    pipeline.add_node(
        name="QuestionNode", component=QuestionOutput(), inputs=["SkQueryKeywordQuestionClassifier.output_1"]
    )
    output = pipeline.run(query="morse code")
    assert output["output"] == "keyword"

    output = pipeline.run(query="How old is John?")
    assert output["output"] == "question"

    pipeline = Pipeline()
    pipeline.add_node(
        name="TfQueryKeywordQuestionClassifier", component=TransformersQueryClassifier(), inputs=["Query"]
    )
    pipeline.add_node(
        name="KeywordNode", component=KeywordOutput(), inputs=["TfQueryKeywordQuestionClassifier.output_2"]
    )
    pipeline.add_node(
        name="QuestionNode", component=QuestionOutput(), inputs=["TfQueryKeywordQuestionClassifier.output_1"]
    )
    output = pipeline.run(query="morse code")
    assert output["output"] == "keyword"

    output = pipeline.run(query="How old is John?")
    assert output["output"] == "question"


@pytest.mark.elasticsearch
@pytest.mark.parametrize("document_store", ["elasticsearch"], indirect=True)
def test_indexing_pipeline_with_classifier(document_store):
    # test correct load of indexing pipeline from yaml
    pipeline = Pipeline.load_from_yaml(
        SAMPLES_PATH / "pipeline" / "test.haystack-pipeline.yml", pipeline_name="indexing_pipeline_with_classifier"
    )
    pipeline.run(file_paths=SAMPLES_PATH / "pdf" / "sample_pdf_1.pdf")
    # test correct load of query pipeline from yaml
    pipeline = Pipeline.load_from_yaml(
        SAMPLES_PATH / "pipeline" / "test.haystack-pipeline.yml", pipeline_name="query_pipeline"
    )
    prediction = pipeline.run(
        query="Who made the PDF specification?", params={"ESRetriever": {"top_k": 10}, "Reader": {"top_k": 3}}
    )
    assert prediction["query"] == "Who made the PDF specification?"
    assert prediction["answers"][0].answer == "Adobe Systems"
    assert prediction["answers"][0].meta["classification"]["label"] == "joy"
    assert "_debug" not in prediction.keys()


@pytest.mark.elasticsearch
@pytest.mark.parametrize("document_store", ["elasticsearch"], indirect=True)
def test_query_pipeline_with_document_classifier(document_store):
    # test correct load of indexing pipeline from yaml
    pipeline = Pipeline.load_from_yaml(
        SAMPLES_PATH / "pipeline" / "test.haystack-pipeline.yml", pipeline_name="indexing_pipeline"
    )
    pipeline.run(file_paths=SAMPLES_PATH / "pdf" / "sample_pdf_1.pdf")
    # test correct load of query pipeline from yaml
    pipeline = Pipeline.load_from_yaml(
        SAMPLES_PATH / "pipeline" / "test.haystack-pipeline.yml",
        pipeline_name="query_pipeline_with_document_classifier",
    )
    prediction = pipeline.run(
        query="Who made the PDF specification?", params={"ESRetriever": {"top_k": 10}, "Reader": {"top_k": 3}}
    )
    assert prediction["query"] == "Who made the PDF specification?"
    assert prediction["answers"][0].answer == "Adobe Systems"
    assert prediction["answers"][0].meta["classification"]["label"] == "joy"
    assert "_debug" not in prediction.keys()
