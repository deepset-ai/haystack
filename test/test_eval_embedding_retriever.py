import pytest
from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack.nodes.retriever.dense import EmbeddingRetriever
from haystack.document_stores.memory import InMemoryDocumentStore
from haystack.nodes.summarizer.transformers import TransformersSummarizer
from haystack.pipelines import GenerativeQAPipeline, SearchSummarizationPipeline
from haystack.schema import EvaluationResult
from test_eval import EVAL_LABELS


# had to be separated from other eval tests to work around OOM in Windows CI

@pytest.mark.parametrize("document_store_with_docs", ["memory"], indirect=True)
@pytest.mark.parametrize("retriever_with_docs", ["embedding"], indirect=True)
def test_generativeqa_calculate_metrics(document_store_with_docs: InMemoryDocumentStore, rag_generator, retriever_with_docs):
    document_store_with_docs.update_embeddings(retriever=retriever_with_docs)
    pipeline = GenerativeQAPipeline(generator=rag_generator, retriever=retriever_with_docs)
    eval_result: EvaluationResult = pipeline.eval(
        labels=EVAL_LABELS,
        params={"Retriever": {"top_k": 5}}
    )

    metrics = eval_result.calculate_metrics()

    assert "Retriever" in eval_result
    assert "Generator" in eval_result
    assert len(eval_result) == 2

    assert metrics["Retriever"]["mrr"] == 0.5
    assert metrics["Retriever"]["map"] == 0.5
    assert metrics["Retriever"]["recall_multi_hit"] == 0.5
    assert metrics["Retriever"]["recall_single_hit"] == 0.5
    assert metrics["Retriever"]["precision"] == 1.0/6
    assert metrics["Generator"]["exact_match"] == 0.0
    assert metrics["Generator"]["f1"] == 1.0/3


@pytest.mark.parametrize("document_store_with_docs", ["memory"], indirect=True)
def test_summarizer_calculate_metrics(document_store_with_docs: ElasticsearchDocumentStore):
    summarizer = TransformersSummarizer(
        model_name_or_path="sshleifer/distill-pegasus-xsum-16-4",
        use_gpu=False
    )
    document_store_with_docs.embedding_dim = 384
    retriever = EmbeddingRetriever(
            document_store=document_store_with_docs,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            use_gpu=False
        )
    document_store_with_docs.update_embeddings(retriever=retriever)
    pipeline = SearchSummarizationPipeline(retriever=retriever, summarizer=summarizer, return_in_answer_format=True)
    eval_result: EvaluationResult = pipeline.eval(
        labels=EVAL_LABELS,
        params={"Retriever": {"top_k": 5}}
    )

    metrics = eval_result.calculate_metrics()

    assert "Retriever" in eval_result
    assert "Summarizer" in eval_result
    assert len(eval_result) == 2

    assert metrics["Retriever"]["mrr"] == 0.5
    assert metrics["Retriever"]["map"] == 0.5
    assert metrics["Retriever"]["recall_multi_hit"] == 0.5
    assert metrics["Retriever"]["recall_single_hit"] == 0.5
    assert metrics["Retriever"]["precision"] == 1.0/6
    assert metrics["Summarizer"]["mrr"] == 0.0
    assert metrics["Summarizer"]["map"] == 0.0
    assert metrics["Summarizer"]["recall_multi_hit"] == 0.0
    assert metrics["Summarizer"]["recall_single_hit"] == 0.0
    assert metrics["Summarizer"]["precision"] == 0.0
