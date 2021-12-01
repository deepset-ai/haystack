import pytest
from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack.nodes.retriever.dense import EmbeddingRetriever
from haystack.document_stores.memory import InMemoryDocumentStore
from haystack.pipelines import ExtractiveQAPipeline, DocumentSearchPipeline, FAQPipeline, GenerativeQAPipeline, SearchSummarizationPipeline
from haystack.pipelines.standard_pipelines import RetrieverQuestionGenerationPipeline, TranslationWrapperPipeline
from haystack.schema import EvaluationResult
from test_eval import EVAL_LABELS, EVAL_QUERIES


@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
@pytest.mark.parametrize("document_store_with_docs", ["memory"], indirect=True)
def test_extractive_qa_eval(reader, retriever_with_docs, tmp_path):
    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever_with_docs)
    eval_result: EvaluationResult = pipeline.eval(
        queries=EVAL_QUERIES, 
        labels=EVAL_LABELS,
        params={"Retriever": {"top_k": 5}}
    )

    metrics = eval_result.calculate_metrics()

    reader_result = eval_result["Reader"]
    retriever_result = eval_result["Retriever"]

    reader_berlin = reader_result[reader_result['query'] == "Who lives in Berlin?"]
    reader_munich = reader_result[reader_result['query'] == "Who lives in Munich?"]

    retriever_berlin = retriever_result[retriever_result['query'] == "Who lives in Berlin?"]
    retriever_munich = retriever_result[retriever_result['query'] == "Who lives in Munich?"]

    assert reader_berlin[reader_berlin['rank'] == 1]["answer"].iloc[0] in reader_berlin[reader_berlin['rank'] == 1]["gold_answers"].iloc[0]
    assert retriever_berlin[retriever_berlin['rank'] == 1]["document_id"].iloc[0] in retriever_berlin[retriever_berlin['rank'] == 1]["gold_document_ids"].iloc[0]
    assert reader_munich[reader_munich['rank'] == 1]["answer"].iloc[0] not in reader_munich[reader_munich['rank'] == 1]["gold_answers"].iloc[0]
    assert retriever_munich[retriever_munich['rank'] == 1]["document_id"].iloc[0] not in retriever_munich[retriever_munich['rank'] == 1]["gold_document_ids"].iloc[0]
    assert metrics["Reader"]["exact_match"] == 1.0
    assert metrics["Reader"]["f1"] == 1.0
    assert metrics["Retriever"]["mrr"] == 0.5
    assert metrics["Retriever"]["map"] == 0.5
    assert metrics["Retriever"]["recall_multi_hit"] == 0.5
    assert metrics["Retriever"]["recall_single_hit"] == 0.5
    assert metrics["Retriever"]["precision"] == 1.0/6

    eval_result.save(tmp_path)
    saved_eval_result = EvaluationResult.load(tmp_path)
    metrics = saved_eval_result.calculate_metrics()

    assert reader_berlin[reader_berlin['rank'] == 1]["answer"].iloc[0] in reader_berlin[reader_berlin['rank'] == 1]["gold_answers"].iloc[0]
    assert retriever_berlin[retriever_berlin['rank'] == 1]["document_id"].iloc[0] in retriever_berlin[retriever_berlin['rank'] == 1]["gold_document_ids"].iloc[0]
    assert reader_munich[reader_munich['rank'] == 1]["answer"].iloc[0] not in reader_munich[reader_munich['rank'] == 1]["gold_answers"].iloc[0]
    assert retriever_munich[retriever_munich['rank'] == 1]["document_id"].iloc[0] not in retriever_munich[retriever_munich['rank'] == 1]["gold_document_ids"].iloc[0]
    assert metrics["Reader"]["exact_match"] == 1.0
    assert metrics["Reader"]["f1"] == 1.0
    assert metrics["Retriever"]["mrr"] == 0.5
    assert metrics["Retriever"]["map"] == 0.5
    assert metrics["Retriever"]["recall_multi_hit"] == 0.5
    assert metrics["Retriever"]["recall_single_hit"] == 0.5
    assert metrics["Retriever"]["precision"] == 1.0/6


@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
@pytest.mark.parametrize("document_store_with_docs", ["memory"], indirect=True)
def test_document_search_calculate_metrics(retriever_with_docs):
    pipeline = DocumentSearchPipeline(retriever=retriever_with_docs)
    eval_result: EvaluationResult = pipeline.eval(
        queries=EVAL_QUERIES, 
        labels=EVAL_LABELS,
        params={"Retriever": {"top_k": 5}}
    )

    metrics = eval_result.calculate_metrics()

    assert "Retriever" in eval_result
    assert len(eval_result) == 1
    retriever_result = eval_result["Retriever"]
    retriever_berlin = retriever_result[retriever_result['query'] == "Who lives in Berlin?"]
    retriever_munich = retriever_result[retriever_result['query'] == "Who lives in Munich?"]

    assert retriever_berlin[retriever_berlin['rank'] == 1]["document_id"].iloc[0] in retriever_berlin[retriever_berlin['rank'] == 1]["gold_document_ids"].iloc[0]
    assert retriever_munich[retriever_munich['rank'] == 1]["document_id"].iloc[0] not in retriever_munich[retriever_munich['rank'] == 1]["gold_document_ids"].iloc[0]
    assert metrics["Retriever"]["mrr"] == 0.5
    assert metrics["Retriever"]["map"] == 0.5
    assert metrics["Retriever"]["recall_multi_hit"] == 0.5
    assert metrics["Retriever"]["recall_single_hit"] == 0.5
    assert metrics["Retriever"]["precision"] == 1.0/6


@pytest.mark.parametrize("document_store_with_docs", ["memory"], indirect=True)
@pytest.mark.parametrize("retriever_with_docs", ["embedding"], indirect=True)
def test_generativeqa_calculate_metrics(document_store_with_docs: InMemoryDocumentStore, rag_generator, retriever_with_docs):
    document_store_with_docs.update_embeddings(retriever=retriever_with_docs)
    pipeline = GenerativeQAPipeline(generator=rag_generator, retriever=retriever_with_docs)
    eval_result: EvaluationResult = pipeline.eval(
        queries=EVAL_QUERIES, 
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
@pytest.mark.parametrize("retriever_with_docs", ["embedding"], indirect=True)
def test_summarizer_calculate_metrics(document_store_with_docs: ElasticsearchDocumentStore, summarizer, retriever_with_docs):
    document_store_with_docs.update_embeddings(retriever=retriever_with_docs)
    pipeline = SearchSummarizationPipeline(retriever=retriever_with_docs, summarizer=summarizer, return_in_answer_format=True)
    eval_result: EvaluationResult = pipeline.eval(
        queries=EVAL_QUERIES, 
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
    assert metrics["Summarizer"]["mrr"] == 0.5
    assert metrics["Summarizer"]["map"] == 0.5
    assert metrics["Summarizer"]["recall_multi_hit"] == 0.5
    assert metrics["Summarizer"]["recall_single_hit"] == 0.5
    assert metrics["Summarizer"]["precision"] == 1.0/6
    

@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
@pytest.mark.parametrize("document_store_with_docs", ["memory"], indirect=True)
def test_faq_calculate_metrics(retriever_with_docs):
    pipeline = FAQPipeline(retriever=retriever_with_docs)
    eval_result: EvaluationResult = pipeline.eval(
        queries=EVAL_QUERIES, 
        labels=EVAL_LABELS,
        params={"Retriever": {"top_k": 5}}
    )

    metrics = eval_result.calculate_metrics()

    assert "Retriever" in eval_result
    assert "Docs2Answers" in eval_result
    assert len(eval_result) == 2

    assert metrics["Retriever"]["mrr"] == 0.5
    assert metrics["Retriever"]["map"] == 0.5
    assert metrics["Retriever"]["recall_multi_hit"] == 0.5
    assert metrics["Retriever"]["recall_single_hit"] == 0.5
    assert metrics["Retriever"]["precision"] == 1.0/6
    assert metrics["Docs2Answers"]["exact_match"] == 0.0
    assert metrics["Docs2Answers"]["f1"] == 0.0


@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
@pytest.mark.parametrize("document_store_with_docs", ["memory"], indirect=True)
def test_extractive_qa_eval_translation(reader, retriever_with_docs, de_to_en_translator):
    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever_with_docs)
    pipeline = TranslationWrapperPipeline(input_translator=de_to_en_translator, output_translator=de_to_en_translator, pipeline=pipeline)
    eval_result: EvaluationResult = pipeline.eval(
        queries=EVAL_QUERIES, 
        labels=EVAL_LABELS,
        params={"Retriever": {"top_k": 5}}
    )

    metrics = eval_result.calculate_metrics()

    assert "Retriever" in eval_result
    assert "Reader" in eval_result
    assert "OutputTranslator" in eval_result
    assert len(eval_result) == 3

    assert metrics["Reader"]["exact_match"] == 1.0
    assert metrics["Reader"]["f1"] == 1.0
    assert metrics["Retriever"]["mrr"] == 0.5
    assert metrics["Retriever"]["map"] == 0.5
    assert metrics["Retriever"]["recall_multi_hit"] == 0.5
    assert metrics["Retriever"]["recall_single_hit"] == 0.5
    assert metrics["Retriever"]["precision"] == 1.0/6

    assert metrics["OutputTranslator"]["exact_match"] == 1.0
    assert metrics["OutputTranslator"]["f1"] == 1.0
    assert metrics["OutputTranslator"]["mrr"] == 0.5
    assert metrics["OutputTranslator"]["map"] == 0.5
    assert metrics["OutputTranslator"]["recall_multi_hit"] == 0.5
    assert metrics["OutputTranslator"]["recall_single_hit"] == 0.5
    assert metrics["OutputTranslator"]["precision"] == 1.0/6


@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
@pytest.mark.parametrize("document_store_with_docs", ["memory"], indirect=True)
def test_question_generation_eval(retriever_with_docs, question_generator):
    pipeline = RetrieverQuestionGenerationPipeline(retriever=retriever_with_docs, question_generator=question_generator)

    eval_result: EvaluationResult = pipeline.eval(
        queries=EVAL_QUERIES, 
        labels=EVAL_LABELS,
        params={"Retriever": {"top_k": 5}}
    )

    metrics = eval_result.calculate_metrics()

    assert "Retriever" in eval_result
    assert "Question Generator" in eval_result
    assert len(eval_result) == 2

    assert metrics["Retriever"]["mrr"] == 0.5
    assert metrics["Retriever"]["map"] == 0.5
    assert metrics["Retriever"]["recall_multi_hit"] == 0.5
    assert metrics["Retriever"]["recall_single_hit"] == 0.5
    assert metrics["Retriever"]["precision"] == 1.0/6

    assert metrics["Question Generator"]["mrr"] == 0.5
    assert metrics["Question Generator"]["map"] == 0.5
    assert metrics["Question Generator"]["recall_multi_hit"] == 0.5
    assert metrics["Question Generator"]["recall_single_hit"] == 0.5
    assert metrics["Question Generator"]["precision"] == 1.0/6
