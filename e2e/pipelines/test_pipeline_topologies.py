import math
import pytest

from haystack.pipelines import Pipeline, RootNode
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import (
    DensePassageRetriever,
    BM25Retriever,
    SklearnQueryClassifier,
    TransformersQueryClassifier,
    JoinDocuments,
    FARMReader,
)


@pytest.mark.parametrize("classifier", [SklearnQueryClassifier(), TransformersQueryClassifier()])
def test_query_keyword_statement_classifier(classifier):
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
    pipeline.add_node(name="classifier", component=classifier, inputs=["Query"])
    pipeline.add_node(name="KeywordNode", component=KeywordOutput(), inputs=["classifier.output_2"])
    pipeline.add_node(name="QuestionNode", component=QuestionOutput(), inputs=["classifier.output_1"])
    output = pipeline.run(query="morse code")
    assert output["output"] == "keyword"

    output = pipeline.run(query="How old is John?")
    assert output["output"] == "question"


def test_join_merge_no_weights(docs):
    document_store = InMemoryDocumentStore(embedding_dim=768, similarity="dot_product", use_bm25=True)
    document_store.write_documents(documents=docs)
    bm25 = BM25Retriever(document_store=document_store)
    dpr = DensePassageRetriever(
        document_store=document_store,
        query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
        passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
        use_gpu=False,
    )
    document_store.update_embeddings(dpr)

    query = "Where does Carla live?"

    join_node = JoinDocuments(join_mode="merge")
    p = Pipeline()
    p.add_node(component=bm25, name="R1", inputs=["Query"])
    p.add_node(component=dpr, name="R2", inputs=["Query"])
    p.add_node(component=join_node, name="Join", inputs=["R1", "R2"])
    results = p.run(query=query)
    assert len(results["documents"]) == 5


def test_join_merge_with_weights(docs):
    document_store = InMemoryDocumentStore(embedding_dim=768, similarity="dot_product", use_bm25=True)
    document_store.write_documents(documents=docs)
    bm25 = BM25Retriever(document_store=document_store)
    dpr = DensePassageRetriever(
        document_store=document_store,
        query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
        passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
        use_gpu=False,
    )
    document_store.update_embeddings(dpr)

    query = "Where does Carla live?"

    join_node = JoinDocuments(join_mode="merge", weights=[1000, 1], top_k_join=2)
    p = Pipeline()
    p.add_node(component=bm25, name="R1", inputs=["Query"])
    p.add_node(component=dpr, name="R2", inputs=["Query"])
    p.add_node(component=join_node, name="Join", inputs=["R1", "R2"])
    results = p.run(query=query)
    assert math.isclose(results["documents"][0].score, 0.5336782589721345, rel_tol=0.0001)
    assert len(results["documents"]) == 2


def test_join_concatenate(docs):
    document_store = InMemoryDocumentStore(embedding_dim=768, similarity="dot_product", use_bm25=True)
    document_store.write_documents(documents=docs)
    bm25 = BM25Retriever(document_store=document_store)
    dpr = DensePassageRetriever(
        document_store=document_store,
        query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
        passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
        use_gpu=False,
    )
    document_store.update_embeddings(dpr)

    query = "Where does Carla live?"

    join_node = JoinDocuments(join_mode="concatenate")
    p = Pipeline()
    p.add_node(component=bm25, name="R1", inputs=["Query"])
    p.add_node(component=dpr, name="R2", inputs=["Query"])
    p.add_node(component=join_node, name="Join", inputs=["R1", "R2"])
    results = p.run(query=query)
    assert len(results["documents"]) == 5


def test_join_concatenate_with_topk(docs):
    document_store = InMemoryDocumentStore(embedding_dim=768, similarity="dot_product", use_bm25=True)
    document_store.write_documents(documents=docs)
    bm25 = BM25Retriever(document_store=document_store)
    dpr = DensePassageRetriever(
        document_store=document_store,
        query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
        passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
        use_gpu=False,
    )
    document_store.update_embeddings(dpr)

    query = "Where does Carla live?"

    join_node = JoinDocuments(join_mode="concatenate")
    p = Pipeline()
    p.add_node(component=bm25, name="R1", inputs=["Query"])
    p.add_node(component=dpr, name="R2", inputs=["Query"])
    p.add_node(component=join_node, name="Join", inputs=["R1", "R2"])
    one_result = p.run(query=query, params={"Join": {"top_k_join": 1}})
    two_results = p.run(query=query, params={"Join": {"top_k_join": 2}})
    assert len(one_result["documents"]) == 1
    assert len(two_results["documents"]) == 2


def test_join_with_reader(docs):
    document_store = InMemoryDocumentStore(embedding_dim=768, similarity="dot_product", use_bm25=True)
    document_store.write_documents(documents=docs)
    bm25 = BM25Retriever(document_store=document_store)
    dpr = DensePassageRetriever(
        document_store=document_store,
        query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
        passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
        use_gpu=False,
    )
    document_store.update_embeddings(dpr)
    reader = FARMReader(
        model_name_or_path="deepset/bert-medium-squad2-distilled", use_gpu=False, top_k_per_sample=5, num_processes=0
    )

    query = "Where does Carla live?"

    join_node = JoinDocuments()
    p = Pipeline()
    p.add_node(component=bm25, name="R1", inputs=["Query"])
    p.add_node(component=dpr, name="R2", inputs=["Query"])
    p.add_node(component=join_node, name="Join", inputs=["R1", "R2"])
    p.add_node(component=reader, name="Reader", inputs=["Join"])
    results = p.run(query=query)
    # check whether correct answer is within top 2 predictions
    assert results["answers"][0].answer == "Berlin" or results["answers"][1].answer == "Berlin"


def test_join_with_rrf(docs):
    document_store = InMemoryDocumentStore(embedding_dim=768, similarity="dot_product", use_bm25=True)
    document_store.write_documents(documents=docs)
    bm25 = BM25Retriever(document_store=document_store)
    dpr = DensePassageRetriever(
        document_store=document_store,
        query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
        passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
        use_gpu=False,
    )
    document_store.update_embeddings(dpr)

    query = "Where does Carla live?"

    join_node = JoinDocuments(join_mode="reciprocal_rank_fusion")
    p = Pipeline()
    p.add_node(component=bm25, name="R1", inputs=["Query"])
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
    assert all(
        doc.score == pytest.approx(expected_scores[idx], abs=1e-3) for idx, doc in enumerate(results["documents"])
    )
