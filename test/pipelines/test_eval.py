from csv import DictWriter
import logging
from pathlib import Path
import pytest
import sys
import pandas as pd
from copy import deepcopy

import responses
from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack.nodes.answer_generator.openai import OpenAIAnswerGenerator
from haystack.nodes.preprocessor import PreProcessor
from haystack.nodes.prompt.prompt_node import PromptNode
from haystack.nodes.query_classifier.transformers import TransformersQueryClassifier
from haystack.nodes.retriever.dense import DensePassageRetriever
from haystack.nodes.retriever.sparse import BM25Retriever
from haystack.nodes.summarizer.transformers import TransformersSummarizer
from haystack.pipelines.base import Pipeline
from haystack.pipelines import ExtractiveQAPipeline, GenerativeQAPipeline, SearchSummarizationPipeline
from haystack.pipelines.standard_pipelines import (
    DocumentSearchPipeline,
    FAQPipeline,
    RetrieverQuestionGenerationPipeline,
    TranslationWrapperPipeline,
)
from haystack.nodes.translator.transformers import TransformersTranslator
from haystack.schema import Answer, Document, EvaluationResult, Label, MultiLabel, Span, TableCell


@pytest.mark.skipif(sys.platform in ["win32", "cygwin"], reason="Causes OOM on windows github runner")
@pytest.mark.parametrize("document_store_with_docs", ["memory"], indirect=True)
@pytest.mark.parametrize("retriever_with_docs", ["embedding"], indirect=True)
def test_summarizer_calculate_metrics(
    document_store_with_docs: ElasticsearchDocumentStore, retriever_with_docs, eval_labels
):
    document_store_with_docs.update_embeddings(retriever=retriever_with_docs)
    summarizer = TransformersSummarizer(model_name_or_path="sshleifer/distill-pegasus-xsum-16-4", use_gpu=False)
    pipeline = SearchSummarizationPipeline(
        retriever=retriever_with_docs, summarizer=summarizer, return_in_answer_format=True
    )
    eval_result: EvaluationResult = pipeline.eval(
        labels=eval_labels, params={"Retriever": {"top_k": 5}}, context_matching_min_length=10
    )

    metrics = eval_result.calculate_metrics(document_scope="context")

    assert "Retriever" in eval_result
    assert "Summarizer" in eval_result
    assert len(eval_result) == 2

    assert metrics["Retriever"]["mrr"] == 1.0
    assert metrics["Retriever"]["map"] == 1.0
    assert metrics["Retriever"]["recall_multi_hit"] == 1.0
    assert metrics["Retriever"]["recall_single_hit"] == 1.0
    assert metrics["Retriever"]["precision"] == 1.0
    assert metrics["Retriever"]["ndcg"] == 1.0
    assert metrics["Summarizer"]["mrr"] == 1.0
    assert metrics["Summarizer"]["map"] == 1.0
    assert metrics["Summarizer"]["recall_multi_hit"] == 1.0
    assert metrics["Summarizer"]["recall_single_hit"] == 1.0
    assert metrics["Summarizer"]["precision"] == 1.0
    assert metrics["Summarizer"]["ndcg"] == 1.0


@pytest.mark.parametrize("document_store", ["elasticsearch", "faiss", "memory"], indirect=True)
@pytest.mark.parametrize("batch_size", [None, 20])
def test_add_eval_data(document_store, batch_size, samples_path):
    # add eval data (SQUAD format)
    document_store.add_eval_data(
        filename=samples_path / "squad" / "small.json",
        doc_index=document_store.index,
        label_index=document_store.label_index,
        batch_size=batch_size,
    )

    assert document_store.get_document_count() == 87
    assert document_store.get_label_count() == 1214

    # test documents
    docs = document_store.get_all_documents(filters={"name": ["Normans"]})
    assert docs[0].meta["name"] == "Normans"
    assert len(docs[0].meta.keys()) == 1

    # test labels
    labels = document_store.get_all_labels()
    label = None
    for l in labels:
        if l.query == "In what country is Normandy located?":
            label = l
            break
    assert label.answer.answer == "France"
    assert label.no_answer == False
    assert label.is_correct_answer == True
    assert label.is_correct_document == True
    assert label.query == "In what country is Normandy located?"
    assert label.origin == "gold-label"
    assert label.answer.offsets_in_document[0].start == 159
    assert (
        label.answer.context[label.answer.offsets_in_context[0].start : label.answer.offsets_in_context[0].end]
        == "France"
    )
    assert label.answer.document_ids == [label.document.id]

    # check combination
    doc = document_store.get_document_by_id(label.document.id)
    start = label.answer.offsets_in_document[0].start
    end = label.answer.offsets_in_document[0].end
    assert end == start + len(label.answer.answer)
    assert doc.content[start:end] == "France"


@pytest.mark.parametrize("document_store", ["elasticsearch", "faiss", "memory"], indirect=True)
@pytest.mark.parametrize("reader", ["farm"], indirect=True)
@pytest.mark.parametrize("use_confidence_scores", [True, False])
def test_eval_reader(reader, document_store, use_confidence_scores, samples_path):
    # add eval data (SQUAD format)
    document_store.add_eval_data(
        filename=samples_path / "squad" / "tiny.json",
        doc_index=document_store.index,
        label_index=document_store.label_index,
    )
    assert document_store.get_document_count() == 2

    reader.use_confidence_scores = use_confidence_scores

    # eval reader
    reader_eval_results = reader.eval(
        document_store=document_store,
        label_index=document_store.label_index,
        doc_index=document_store.index,
        device="cpu",
    )

    if use_confidence_scores:
        assert reader_eval_results["f1"] == 50
        assert reader_eval_results["EM"] == 50
        assert reader_eval_results["top_n_accuracy"] == 100.0
    else:
        assert reader_eval_results["f1"] == 50
        assert reader_eval_results["EM"] == 50
        assert reader_eval_results["top_n_accuracy"] == 100.0


# using ElasticsearchDocumentStore, since InMemoryDocumentStore doesn't return meaningful BM25 scores when there are very few documents
@pytest.mark.elasticsearch
@pytest.mark.parametrize("document_store", ["elasticsearch"], indirect=True)
@pytest.mark.parametrize("open_domain", [True, False])
@pytest.mark.parametrize("retriever", ["bm25"], indirect=True)
def test_eval_elastic_retriever(document_store, open_domain, retriever, samples_path):
    # add eval data (SQUAD format)
    document_store.add_eval_data(
        filename=samples_path / "squad" / "tiny.json",
        doc_index=document_store.index,
        label_index=document_store.label_index,
    )
    assert document_store.get_document_count() == 2

    # eval retriever
    results = retriever.eval(
        top_k=1, label_index=document_store.label_index, doc_index=document_store.index, open_domain=open_domain
    )
    assert results["recall"] == 1.0
    assert results["mrr"] == 1.0
    if not open_domain:
        assert results["map"] == 1.0


@pytest.mark.parametrize("document_store", ["memory"], indirect=True)
@pytest.mark.parametrize("reader", ["farm"], indirect=True)
@pytest.mark.parametrize("retriever", ["bm25"], indirect=True)
def test_eval_pipeline(document_store, reader, retriever, samples_path):
    # add eval data (SQUAD format)
    document_store.add_eval_data(
        filename=samples_path / "squad" / "tiny.json",
        doc_index=document_store.index,
        label_index=document_store.label_index,
    )
    assert document_store.get_document_count() == 2

    p = Pipeline()
    p.add_node(component=retriever, name="Retriever", inputs=["Query"])
    p.add_node(component=reader, name="Reader", inputs=["Retriever"])

    labels = document_store.get_all_labels_aggregated(drop_negative_labels=True, drop_no_answers=False)

    metrics_vanilla = p.eval(labels=labels, params={"Retriever": {"top_k": 5}}).calculate_metrics()
    metrics_sas_sentence_transformers = p.eval(
        labels=labels,
        params={"Retriever": {"top_k": 5}},
        sas_model_name_or_path="sentence-transformers/paraphrase-MiniLM-L3-v2",
    ).calculate_metrics()
    metrics_sas_cross_encoder = p.eval(
        labels=labels, params={"Retriever": {"top_k": 5}}, sas_model_name_or_path="cross-encoder/stsb-TinyBERT-L-4"
    ).calculate_metrics()

    assert metrics_vanilla["Retriever"]["recall_single_hit"] == 1.0
    assert metrics_sas_sentence_transformers["Reader"]["f1"] == pytest.approx(0.75)
    assert metrics_sas_sentence_transformers["Reader"]["exact_match"] == 0.5
    assert metrics_sas_sentence_transformers["Reader"]["sas"] == pytest.approx(0.87586, 1e-4)
    assert metrics_sas_sentence_transformers["Reader"]["exact_match"] == metrics_vanilla["Reader"]["exact_match"]
    assert metrics_sas_cross_encoder["Reader"]["sas"] == pytest.approx(0.71063, 1e-4)


@pytest.mark.parametrize("document_store", ["elasticsearch", "faiss", "memory"], indirect=True)
def test_eval_data_split_word(document_store, samples_path):
    # splitting by word
    preprocessor = PreProcessor(
        clean_empty_lines=False,
        clean_whitespace=False,
        clean_header_footer=False,
        split_by="word",
        split_length=4,
        split_overlap=0,
        split_respect_sentence_boundary=False,
    )

    document_store.add_eval_data(
        filename=samples_path / "squad" / "tiny.json",
        doc_index=document_store.index,
        label_index=document_store.label_index,
        preprocessor=preprocessor,
    )
    labels = document_store.get_all_labels_aggregated()
    docs = document_store.get_all_documents()
    assert len(docs) == 5
    assert len(set(labels[0].document_ids)) == 2


@pytest.mark.parametrize("document_store", ["elasticsearch", "faiss", "memory"], indirect=True)
def test_eval_data_split_passage(document_store, samples_path):
    # splitting by passage
    preprocessor = PreProcessor(
        clean_empty_lines=False,
        clean_whitespace=False,
        clean_header_footer=False,
        split_by="passage",
        split_length=1,
        split_overlap=0,
        split_respect_sentence_boundary=False,
    )

    document_store.add_eval_data(
        filename=samples_path / "squad" / "tiny_passages.json",
        doc_index=document_store.index,
        label_index=document_store.label_index,
        preprocessor=preprocessor,
    )
    docs = document_store.get_all_documents()
    assert len(docs) == 2
    assert len(docs[1].content) == 56


NO_ANSWER_EVAL_LABELS = [
    MultiLabel(
        labels=[
            Label(
                query="Why does probability work?",
                document=Document(""),
                answer=None,
                is_correct_answer=True,
                is_correct_document=True,
                origin="gold-label",
            )
        ]
    )
]

DOC_SEARCH_EVAL_LABELS = [
    MultiLabel(
        labels=[
            Label(
                query="Who lives in Berlin?",
                answer=None,
                document=Document(
                    id="a0747b83aea0b60c4b114b15476dd32d",
                    content_type="text",
                    content="My name is Carla and I live in Berlin",
                ),
                is_correct_answer=False,
                is_correct_document=True,
                origin="gold-label",
            )
        ]
    ),
    MultiLabel(
        labels=[
            Label(
                query="Who lives in Munich?",
                answer=None,
                document=Document(
                    id="something_else", content_type="text", content="My name is Carla and I live in Munich"
                ),
                is_correct_answer=False,
                is_correct_document=True,
                origin="gold-label",
            )
        ]
    ),
]

DOC_SEARCH_ID_EVAL_LABELS = [
    MultiLabel(
        labels=[
            Label(
                query="Who lives in Berlin?",
                answer=None,
                document=Document(id="a0747b83aea0b60c4b114b15476dd32d", content_type="text", content=""),
                is_correct_answer=False,
                is_correct_document=True,
                origin="gold-label",
            )
        ]
    ),
    MultiLabel(
        labels=[
            Label(
                query="Who lives in Munich?",
                answer=None,
                document=Document(id="something_else", content_type="text", content=""),
                is_correct_answer=False,
                is_correct_document=True,
                origin="gold-label",
            )
        ]
    ),
]

FILE_SEARCH_EVAL_LABELS = [
    MultiLabel(
        labels=[
            Label(
                query="Who lives in Berlin?",
                answer=None,
                document=Document(content_type="text", content="", meta={"name": "filename1"}),
                is_correct_answer=False,
                is_correct_document=True,
                origin="gold-label",
            )
        ]
    ),
    MultiLabel(
        labels=[
            Label(
                query="Who lives in Munich?",
                answer=None,
                document=Document(content_type="text", content="", meta={"name": "filename2"}),
                is_correct_answer=False,
                is_correct_document=True,
                origin="gold-label",
            )
        ]
    ),
]

EVAL_TABLE_LABELS = [
    MultiLabel(
        labels=[
            Label(
                query="How old is Brad Pitt?",
                answer=Answer(answer="56", offsets_in_context=[TableCell(1, 2)]),
                document=Document(
                    id="a044cf3fb8aade03a12399c7a2fe9a6b",
                    content_type="table",
                    content=pd.DataFrame(
                        columns=["Actors", "Age", "Number of movies"],
                        data=[
                            ["Brad Pitt", "56", "87"],
                            ["Leonardo Di Caprio", "45", "53"],
                            ["George Clooney", "59", "69"],
                        ],
                    ),
                ),
                is_correct_answer=True,
                is_correct_document=True,
                origin="gold-label",
            ),
            Label(  # Label with different doc but same answer and query
                query="How old is Brad Pitt?",
                answer=Answer(answer="56", offsets_in_context=[TableCell(4, 5)]),
                document=Document(
                    id="a044cf3fb8aade03a12399c7a2fe9a6b",
                    content_type="table",
                    content=pd.DataFrame(
                        columns=["Actors", "Age", "Number of movies"],
                        data=[["Beyonce", "45", "53"], ["Brad Pitt", "56", "87"], ["Jane Doe", "59", "69"]],
                    ),
                ),
                is_correct_answer=True,
                is_correct_document=True,
                origin="gold-label",
            ),
        ]
    ),
    MultiLabel(
        labels=[
            Label(
                query="To which state does Spikeroog belong?",
                answer=Answer(answer="Lower Saxony", offsets_in_context=[TableCell(7, 8)]),
                document=Document(
                    id="b044cf3fb8aade03a12399c7a2fe9a6c",
                    content_type="table",
                    content=pd.DataFrame(
                        columns=["0", "1"],
                        data=[
                            ["Area", "18.25 km2 (7.05 sq mi)"],
                            ["Population", "794"],
                            ["Country", "Germany"],
                            ["State", "Lower Saxony"],
                            ["District", "Wittmund"],
                        ],
                    ),
                ),
                is_correct_answer=True,
                is_correct_document=True,
                origin="gold-label",
            )
        ]
    ),
]


@pytest.mark.skip(reason="Should be an end-to-end test since it uses model inferencing")
@pytest.mark.integration
@pytest.mark.parametrize("document_store", ["memory"], indirect=True)
@pytest.mark.parametrize("retriever", ["table_text_retriever"], indirect=True)
@pytest.mark.parametrize("table_reader_and_param", ["tapas_small"], indirect=True)
@pytest.mark.embedding_dim(512)
def test_table_qa_eval(table_reader_and_param, document_store, retriever):
    docs = []
    for multi_label in EVAL_TABLE_LABELS:
        for label in multi_label.labels:
            docs.append(label.document)

    assert len(docs) == 3

    document_store.write_documents(docs)
    document_store.update_embeddings(retriever=retriever)

    table_reader, _ = table_reader_and_param
    p = Pipeline()
    p.add_node(component=retriever, name="TableRetriever", inputs=["Query"])
    p.add_node(component=table_reader, name="TableReader", inputs=["TableRetriever"])

    eval_result = p.eval(labels=EVAL_TABLE_LABELS, params={"TableRetriever": {"top_k": 2}})
    table_reader_results = eval_result.node_results["TableReader"]

    assert set(table_reader_results["query"].tolist()) == {
        "How old is Brad Pitt?",
        "To which state does Spikeroog belong?",
    }

    metrics = eval_result.calculate_metrics(document_scope="document_id_or_answer")
    assert metrics["TableRetriever"]["recall_single_hit"] == 1.0
    assert metrics["TableRetriever"]["recall_multi_hit"] == 1.0
    assert metrics["TableRetriever"]["precision"] == 0.5
    assert metrics["TableRetriever"]["mrr"] == 1.0
    assert metrics["TableRetriever"]["map"] == 1.0
    assert metrics["TableRetriever"]["ndcg"] == 1.0
    assert metrics["TableReader"]["exact_match"] == 1.0
    assert metrics["TableReader"]["f1"] == 1.0

    # assert metrics are floats
    for node_metrics in metrics.values():
        for value in node_metrics.values():
            assert isinstance(value, float)


@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
@pytest.mark.parametrize("document_store_with_docs", ["memory"], indirect=True)
@pytest.mark.parametrize("reader", ["farm"], indirect=True)
def test_extractive_qa_eval(reader, retriever_with_docs, tmp_path, eval_labels):
    labels = eval_labels[:1]

    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever_with_docs)
    eval_result = pipeline.eval(labels=labels, params={"Retriever": {"top_k": 5}})

    metrics = eval_result.calculate_metrics(document_scope="document_id")

    reader_result = eval_result["Reader"]
    retriever_result = eval_result["Retriever"]

    expected_reader_result_columns = [
        "answer",  # answer-specific
        "exact_match",  # answer-specific
        "f1",  # answer-specific
        # "sas",  # answer-specific optional
        "exact_match_context_scope",  # answer-specific
        "f1_context_scope",  # answer-specific
        # "sas_context_scope",  # answer-specific optional
        "exact_match_document_id_scope",  # answer-specific
        "f1_document_id_scope",  # answer-specific
        # "sas_document_id_scope",  # answer-specific optional
        "exact_match_document_id_and_context_scope",  # answer-specific
        "f1_document_id_and_context_scope",  # answer-specific
        # "sas_document_id_and_context_scope",  # answer-specific optional
        "offsets_in_document",  # answer-specific
        "gold_offsets_in_documents",  # answer-specific
        "offsets_in_context",  # answer-specific
        "gold_offsets_in_contexts",  # answer-specific
        "gold_answers_exact_match",  # answer-specific
        "gold_answers_f1",  # answer-specific
        # "gold_answers_sas",  # answer-specific optional
        "document_ids",  # answer-specific
    ]

    expected_retriever_result_columns = [
        "gold_id_match",  # doc-specific
        "context_match",  # doc-specific
        "answer_match",  # doc-specific
        "gold_id_or_answer_match",  # doc-specific
        "gold_id_and_answer_match",  # doc-specific
        "gold_id_or_context_match",  # doc-specific
        "gold_id_and_context_match",  # doc-specific
        "gold_id_and_context_and_answer_match",  # doc-specific
        "context_and_answer_match",  # doc-specific
        "gold_answers_match",  # doc-specific,
        "document_id",  # doc-specific
    ]

    expected_generic_result_columns = [
        "multilabel_id",  # generic
        "query",  # generic
        "filters",  # generic
        "context",  # generic
        "gold_contexts",  # generic
        "gold_documents_id_match",  # generic
        "gold_contexts_similarity",  # generic
        "type",  # generic
        "node",  # generic
        "eval_mode",  # generic
        "rank",  # generic
        "gold_document_ids",  # generic
        "gold_answers",  # generic
        # "custom_document_id",  # generic optional
        # "gold_custom_document_ids",  # generic optional
    ]

    # all expected columns are part of the evaluation result dataframe
    assert sorted(expected_reader_result_columns + expected_generic_result_columns + ["index"]) == sorted(
        list(reader_result.columns)
    )
    assert sorted(expected_retriever_result_columns + expected_generic_result_columns + ["index"]) == sorted(
        list(retriever_result.columns)
    )

    assert (
        reader_result[reader_result["rank"] == 1]["answer"].iloc[0]
        in reader_result[reader_result["rank"] == 1]["gold_answers"].iloc[0]
    )
    assert (
        retriever_result[retriever_result["rank"] == 1]["document_id"].iloc[0]
        in retriever_result[retriever_result["rank"] == 1]["gold_document_ids"].iloc[0]
    )
    assert metrics["Reader"]["exact_match"] == 1.0
    assert metrics["Reader"]["f1"] == 1.0
    assert metrics["Retriever"]["mrr"] == 1.0
    assert metrics["Retriever"]["recall_multi_hit"] == 1.0
    assert metrics["Retriever"]["recall_single_hit"] == 1.0
    assert metrics["Retriever"]["precision"] == 0.2
    assert metrics["Retriever"]["map"] == 1.0
    assert metrics["Retriever"]["ndcg"] == 1.0

    # assert metrics are floats
    for node_metrics in metrics.values():
        for value in node_metrics.values():
            assert isinstance(value, float)

    eval_result.save(tmp_path)
    saved_eval_result = EvaluationResult.load(tmp_path)

    for key, df in eval_result.node_results.items():
        pd.testing.assert_frame_equal(df, saved_eval_result[key])

    metrics = saved_eval_result.calculate_metrics(document_scope="document_id")

    assert (
        reader_result[reader_result["rank"] == 1]["answer"].iloc[0]
        in reader_result[reader_result["rank"] == 1]["gold_answers"].iloc[0]
    )
    assert (
        retriever_result[retriever_result["rank"] == 1]["document_id"].iloc[0]
        in retriever_result[retriever_result["rank"] == 1]["gold_document_ids"].iloc[0]
    )
    assert metrics["Reader"]["exact_match"] == 1.0
    assert metrics["Reader"]["f1"] == 1.0
    assert metrics["Retriever"]["mrr"] == 1.0
    assert metrics["Retriever"]["recall_multi_hit"] == 1.0
    assert metrics["Retriever"]["recall_single_hit"] == 1.0
    assert metrics["Retriever"]["precision"] == 0.2
    assert metrics["Retriever"]["map"] == 1.0
    assert metrics["Retriever"]["ndcg"] == 1.0

    # assert metrics are floats
    for node_metrics in metrics.values():
        for value in node_metrics.values():
            assert isinstance(value, float)


@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
@pytest.mark.parametrize("document_store_with_docs", ["memory"], indirect=True)
@responses.activate
def test_generative_qa_eval(retriever_with_docs, tmp_path, eval_labels):
    labels = eval_labels[:1]
    responses.add(
        responses.POST,
        "https://api.openai.com/v1/completions",
        json={"choices": [{"text": "test", "finish_reason": "stop"}, {"text": "test2", "finish_reason": "stop"}]},
        status=200,
    )
    responses.add_passthru("https://openaipublic.blob.core.windows.net")
    generator = OpenAIAnswerGenerator(api_key="dummy", top_k=2)
    pipeline = GenerativeQAPipeline(generator=generator, retriever=retriever_with_docs)
    eval_result = pipeline.eval(labels=labels, params={"Retriever": {"top_k": 5}})

    metrics = eval_result.calculate_metrics(document_scope="document_id")

    generator_result = eval_result["Generator"]
    retriever_result = eval_result["Retriever"]

    expected_generator_result_columns = [
        "answer",  # answer-specific
        "exact_match",  # answer-specific
        "f1",  # answer-specific
        # "sas",  # answer-specific optional
        "exact_match_context_scope",  # answer-specific
        "f1_context_scope",  # answer-specific
        # "sas_context_scope",  # answer-specific optional
        "exact_match_document_id_scope",  # answer-specific
        "f1_document_id_scope",  # answer-specific
        # "sas_document_id_scope",  # answer-specific optional
        "exact_match_document_id_and_context_scope",  # answer-specific
        "f1_document_id_and_context_scope",  # answer-specific
        # "sas_document_id_and_context_scope",  # answer-specific optional
        "offsets_in_document",  # answer-specific
        "gold_offsets_in_documents",  # answer-specific
        "offsets_in_context",  # answer-specific
        "gold_offsets_in_contexts",  # answer-specific
        "gold_answers_exact_match",  # answer-specific
        "gold_answers_f1",  # answer-specific
        # "gold_answers_sas",  # answer-specific optional
        "document_ids",  # answer-specific
        "prompt",  # answer-specific
    ]

    expected_retriever_result_columns = [
        "gold_id_match",  # doc-specific
        "context_match",  # doc-specific
        "answer_match",  # doc-specific
        "gold_id_or_answer_match",  # doc-specific
        "gold_id_and_answer_match",  # doc-specific
        "gold_id_or_context_match",  # doc-specific
        "gold_id_and_context_match",  # doc-specific
        "gold_id_and_context_and_answer_match",  # doc-specific
        "context_and_answer_match",  # doc-specific
        "gold_answers_match",  # doc-specific,
        "document_id",  # doc-specific
    ]

    expected_generic_result_columns = [
        "multilabel_id",  # generic
        "query",  # generic
        "filters",  # generic
        "context",  # generic
        "gold_contexts",  # generic
        "gold_documents_id_match",  # generic
        "gold_contexts_similarity",  # generic
        "type",  # generic
        "node",  # generic
        "eval_mode",  # generic
        "rank",  # generic
        "gold_document_ids",  # generic
        "gold_answers",  # generic
        # "custom_document_id",  # generic optional
        # "gold_custom_document_ids",  # generic optional
    ]

    # all expected columns are part of the evaluation result dataframe
    assert sorted(expected_generator_result_columns + expected_generic_result_columns + ["index"]) == sorted(
        list(generator_result.columns)
    )
    assert sorted(expected_retriever_result_columns + expected_generic_result_columns + ["index"]) == sorted(
        list(retriever_result.columns)
    )

    assert generator_result["prompt"].iloc[0] is not None

    # assert metrics are floats
    for node_metrics in metrics.values():
        for value in node_metrics.values():
            assert isinstance(value, float)

    eval_result.save(tmp_path)
    saved_eval_result = EvaluationResult.load(tmp_path)

    for key, df in eval_result.node_results.items():
        pd.testing.assert_frame_equal(df, saved_eval_result[key])

    loaded_metrics = saved_eval_result.calculate_metrics(document_scope="document_id")
    assert metrics == loaded_metrics


@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
@pytest.mark.parametrize("document_store_with_docs", ["memory"], indirect=True)
def test_generative_qa_w_promptnode_eval(retriever_with_docs, tmp_path, eval_labels):
    labels = eval_labels[:1]
    pipeline = Pipeline()
    pipeline.add_node(retriever_with_docs, name="Retriever", inputs=["Query"])
    pipeline.add_node(
        PromptNode(default_prompt_template="question-answering", model_name_or_path="google/flan-t5-small", top_k=2),
        name="PromptNode",
        inputs=["Retriever"],
    )

    eval_result = pipeline.eval(labels=labels, params={"Retriever": {"top_k": 5}})

    metrics = eval_result.calculate_metrics(document_scope="document_id")

    generator_result = eval_result["PromptNode"]
    retriever_result = eval_result["Retriever"]

    expected_generator_result_columns = [
        "answer",  # answer-specific
        "exact_match",  # answer-specific
        "f1",  # answer-specific
        # "sas",  # answer-specific optional
        "exact_match_context_scope",  # answer-specific
        "f1_context_scope",  # answer-specific
        # "sas_context_scope",  # answer-specific optional
        "exact_match_document_id_scope",  # answer-specific
        "f1_document_id_scope",  # answer-specific
        # "sas_document_id_scope",  # answer-specific optional
        "exact_match_document_id_and_context_scope",  # answer-specific
        "f1_document_id_and_context_scope",  # answer-specific
        # "sas_document_id_and_context_scope",  # answer-specific optional
        "offsets_in_document",  # answer-specific
        "gold_offsets_in_documents",  # answer-specific
        "offsets_in_context",  # answer-specific
        "gold_offsets_in_contexts",  # answer-specific
        "gold_answers_exact_match",  # answer-specific
        "gold_answers_f1",  # answer-specific
        # "gold_answers_sas",  # answer-specific optional
        "document_ids",  # answer-specific
        "prompt",  # answer-specific
    ]

    expected_retriever_result_columns = [
        "gold_id_match",  # doc-specific
        "context_match",  # doc-specific
        "answer_match",  # doc-specific
        "gold_id_or_answer_match",  # doc-specific
        "gold_id_and_answer_match",  # doc-specific
        "gold_id_or_context_match",  # doc-specific
        "gold_id_and_context_match",  # doc-specific
        "gold_id_and_context_and_answer_match",  # doc-specific
        "context_and_answer_match",  # doc-specific
        "gold_answers_match",  # doc-specific,
        "document_id",  # doc-specific
    ]

    expected_generic_result_columns = [
        "multilabel_id",  # generic
        "query",  # generic
        "filters",  # generic
        "context",  # generic
        "gold_contexts",  # generic
        "gold_documents_id_match",  # generic
        "gold_contexts_similarity",  # generic
        "type",  # generic
        "node",  # generic
        "eval_mode",  # generic
        "rank",  # generic
        "gold_document_ids",  # generic
        "gold_answers",  # generic
        # "custom_document_id",  # generic optional
        # "gold_custom_document_ids",  # generic optional
    ]

    # all expected columns are part of the evaluation result dataframe
    assert sorted(expected_generator_result_columns + expected_generic_result_columns + ["index"]) == sorted(
        list(generator_result.columns)
    )
    assert sorted(expected_retriever_result_columns + expected_generic_result_columns + ["index"]) == sorted(
        list(retriever_result.columns)
    )

    assert generator_result["prompt"].iloc[0] is not None

    # assert metrics are floats
    for node_metrics in metrics.values():
        for value in node_metrics.values():
            assert isinstance(value, float)

    eval_result.save(tmp_path)
    saved_eval_result = EvaluationResult.load(tmp_path)

    for key, df in eval_result.node_results.items():
        pd.testing.assert_frame_equal(df, saved_eval_result[key])

    loaded_metrics = saved_eval_result.calculate_metrics(document_scope="document_id")
    assert metrics == loaded_metrics


@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
@pytest.mark.parametrize("document_store_with_docs", ["memory"], indirect=True)
@pytest.mark.parametrize("reader", ["farm"], indirect=True)
def test_extractive_qa_eval_multiple_queries(reader, retriever_with_docs, tmp_path, eval_labels):
    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever_with_docs)
    eval_result: EvaluationResult = pipeline.eval(labels=eval_labels, params={"Retriever": {"top_k": 5}})

    metrics = eval_result.calculate_metrics(document_scope="document_id")

    reader_result = eval_result["Reader"]
    retriever_result = eval_result["Retriever"]

    reader_berlin = reader_result[reader_result["query"] == "Who lives in Berlin?"]
    reader_munich = reader_result[reader_result["query"] == "Who lives in Munich?"]

    retriever_berlin = retriever_result[retriever_result["query"] == "Who lives in Berlin?"]
    retriever_munich = retriever_result[retriever_result["query"] == "Who lives in Munich?"]

    assert (
        reader_berlin[reader_berlin["rank"] == 1]["answer"].iloc[0]
        in reader_berlin[reader_berlin["rank"] == 1]["gold_answers"].iloc[0]
    )
    assert (
        retriever_berlin[retriever_berlin["rank"] == 1]["document_id"].iloc[0]
        in retriever_berlin[retriever_berlin["rank"] == 1]["gold_document_ids"].iloc[0]
    )
    assert (
        reader_munich[reader_munich["rank"] == 1]["answer"].iloc[0]
        not in reader_munich[reader_munich["rank"] == 1]["gold_answers"].iloc[0]
    )
    assert (
        retriever_munich[retriever_munich["rank"] == 1]["document_id"].iloc[0]
        not in retriever_munich[retriever_munich["rank"] == 1]["gold_document_ids"].iloc[0]
    )
    assert metrics["Reader"]["exact_match"] == 1.0
    assert metrics["Reader"]["f1"] == 1.0
    assert metrics["Retriever"]["mrr"] == 0.5
    assert metrics["Retriever"]["map"] == 0.5
    assert metrics["Retriever"]["recall_multi_hit"] == 0.5
    assert metrics["Retriever"]["recall_single_hit"] == 0.5
    assert metrics["Retriever"]["precision"] == 0.1
    assert metrics["Retriever"]["ndcg"] == 0.5

    eval_result.save(tmp_path)
    saved_eval_result = EvaluationResult.load(tmp_path)

    for key, df in eval_result.node_results.items():
        pd.testing.assert_frame_equal(df, saved_eval_result[key])

    metrics = saved_eval_result.calculate_metrics(document_scope="document_id")

    assert (
        reader_berlin[reader_berlin["rank"] == 1]["answer"].iloc[0]
        in reader_berlin[reader_berlin["rank"] == 1]["gold_answers"].iloc[0]
    )
    assert (
        retriever_berlin[retriever_berlin["rank"] == 1]["document_id"].iloc[0]
        in retriever_berlin[retriever_berlin["rank"] == 1]["gold_document_ids"].iloc[0]
    )
    assert (
        reader_munich[reader_munich["rank"] == 1]["answer"].iloc[0]
        not in reader_munich[reader_munich["rank"] == 1]["gold_answers"].iloc[0]
    )
    assert (
        retriever_munich[retriever_munich["rank"] == 1]["document_id"].iloc[0]
        not in retriever_munich[retriever_munich["rank"] == 1]["gold_document_ids"].iloc[0]
    )
    assert metrics["Reader"]["exact_match"] == 1.0
    assert metrics["Reader"]["f1"] == 1.0
    assert metrics["Retriever"]["mrr"] == 0.5
    assert metrics["Retriever"]["map"] == 0.5
    assert metrics["Retriever"]["recall_multi_hit"] == 0.5
    assert metrics["Retriever"]["recall_single_hit"] == 0.5
    assert metrics["Retriever"]["precision"] == 0.1
    assert metrics["Retriever"]["ndcg"] == 0.5


@pytest.mark.parametrize("retriever_with_docs", ["bm25"], indirect=True)
@pytest.mark.parametrize("document_store_with_docs", ["elasticsearch"], indirect=True)
@pytest.mark.parametrize("reader", ["farm"], indirect=True)
def test_extractive_qa_labels_with_filters(reader, retriever_with_docs, tmp_path):
    labels = [
        # MultiLabel with filter that selects only the document about Carla
        MultiLabel(
            labels=[
                Label(
                    query="What's her name?",
                    answer=Answer(answer="Carla", offsets_in_context=[Span(11, 16)]),
                    document=Document(
                        id="a0747b83aea0b60c4b114b15476dd32d",
                        content_type="text",
                        content="My name is Carla and I live in Berlin",
                    ),
                    is_correct_answer=True,
                    is_correct_document=True,
                    origin="gold-label",
                    filters={"name": ["filename1"]},
                )
            ]
        ),
        # MultiLabel with filter that selects only the document about Christelle
        MultiLabel(
            labels=[
                Label(
                    query="What's her name?",
                    answer=Answer(answer="Christelle", offsets_in_context=[Span(11, 20)]),
                    document=Document(
                        id="4fa3938bef1d83e4d927669666d0b705",
                        content_type="text",
                        content="My name is Christelle and I live in Paris",
                    ),
                    is_correct_answer=True,
                    is_correct_document=True,
                    origin="gold-label",
                    filters={"name": ["filename3"]},
                )
            ]
        ),
    ]

    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever_with_docs)
    eval_result = pipeline.eval(labels=labels, params={"Retriever": {"top_k": 5}})

    metrics = eval_result.calculate_metrics(document_scope="document_id")

    reader_result = eval_result["Reader"]
    retriever_result = eval_result["Retriever"]

    # The same query but with two different filters and thus two different answers is answered correctly in both cases.
    assert (
        reader_result[reader_result["rank"] == 1]["answer"].iloc[0]
        in reader_result[reader_result["rank"] == 1]["gold_answers"].iloc[0]
    )
    assert (
        retriever_result[retriever_result["rank"] == 1]["document_id"].iloc[0]
        in retriever_result[retriever_result["rank"] == 1]["gold_document_ids"].iloc[0]
    )
    assert metrics["Reader"]["exact_match"] == 1.0
    assert metrics["Reader"]["f1"] == 1.0
    assert metrics["Retriever"]["mrr"] == 1.0
    assert metrics["Retriever"]["recall_multi_hit"] == 1.0
    assert metrics["Retriever"]["recall_single_hit"] == 1.0
    assert metrics["Retriever"]["precision"] == 1.0
    assert metrics["Retriever"]["map"] == 1.0
    assert metrics["Retriever"]["ndcg"] == 1.0


@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
@pytest.mark.parametrize("document_store_with_docs", ["memory"], indirect=True)
@pytest.mark.parametrize("reader", ["farm"], indirect=True)
def test_extractive_qa_eval_sas(reader, retriever_with_docs, eval_labels):
    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever_with_docs)
    eval_result: EvaluationResult = pipeline.eval(
        labels=eval_labels,
        params={"Retriever": {"top_k": 5}},
        sas_model_name_or_path="sentence-transformers/paraphrase-MiniLM-L3-v2",
    )

    metrics = eval_result.calculate_metrics(document_scope="document_id")

    assert metrics["Reader"]["exact_match"] == 1.0
    assert metrics["Reader"]["f1"] == 1.0
    assert metrics["Retriever"]["mrr"] == 0.5
    assert metrics["Retriever"]["map"] == 0.5
    assert metrics["Retriever"]["recall_multi_hit"] == 0.5
    assert metrics["Retriever"]["recall_single_hit"] == 0.5
    assert metrics["Retriever"]["precision"] == 0.1
    assert metrics["Retriever"]["ndcg"] == 0.5
    assert "sas" in metrics["Reader"]
    assert metrics["Reader"]["sas"] == pytest.approx(1.0)

    # assert metrics are floats
    for node_metrics in metrics.values():
        for value in node_metrics.values():
            assert isinstance(value, float)


@pytest.mark.parametrize("reader", ["farm"], indirect=True)
def test_reader_eval_in_pipeline(reader, eval_labels):
    pipeline = Pipeline()
    pipeline.add_node(component=reader, name="Reader", inputs=["Query"])
    eval_result: EvaluationResult = pipeline.eval(
        labels=eval_labels,
        documents=[[label.document for label in multilabel.labels] for multilabel in eval_labels],
        params={},
    )

    metrics = eval_result.calculate_metrics(document_scope="document_id")

    assert metrics["Reader"]["exact_match"] == 1.0
    assert metrics["Reader"]["f1"] == 1.0


@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
@pytest.mark.parametrize("document_store_with_docs", ["memory"], indirect=True)
def test_extractive_qa_eval_document_scope(retriever_with_docs, eval_labels):
    pipeline = DocumentSearchPipeline(retriever=retriever_with_docs)
    eval_result: EvaluationResult = pipeline.eval(
        labels=eval_labels,
        params={"Retriever": {"top_k": 5}},
        context_matching_min_length=20,  # artificially set down min_length to see if context matching is working properly
    )

    metrics = eval_result.calculate_metrics(document_scope="document_id")

    assert metrics["Retriever"]["mrr"] == 0.5
    assert metrics["Retriever"]["map"] == 0.5
    assert metrics["Retriever"]["recall_multi_hit"] == 0.5
    assert metrics["Retriever"]["recall_single_hit"] == 0.5
    assert metrics["Retriever"]["precision"] == 0.1
    assert metrics["Retriever"]["ndcg"] == 0.5

    metrics = eval_result.calculate_metrics(document_scope="context")

    assert metrics["Retriever"]["mrr"] == 1.0
    assert metrics["Retriever"]["map"] == 1.0
    assert metrics["Retriever"]["recall_multi_hit"] == 1.0
    assert metrics["Retriever"]["recall_single_hit"] == 1.0
    assert metrics["Retriever"]["precision"] == 1.0
    assert metrics["Retriever"]["ndcg"] == 1.0

    metrics = eval_result.calculate_metrics(document_scope="document_id_and_context")

    assert metrics["Retriever"]["mrr"] == 0.5
    assert metrics["Retriever"]["map"] == 0.5
    assert metrics["Retriever"]["recall_multi_hit"] == 0.5
    assert metrics["Retriever"]["recall_single_hit"] == 0.5
    assert metrics["Retriever"]["precision"] == 0.1
    assert metrics["Retriever"]["ndcg"] == 0.5

    metrics = eval_result.calculate_metrics(document_scope="document_id_or_context")

    assert metrics["Retriever"]["mrr"] == 1.0
    assert metrics["Retriever"]["map"] == 1.0
    assert metrics["Retriever"]["recall_multi_hit"] == 1.0
    assert metrics["Retriever"]["recall_single_hit"] == 1.0
    assert metrics["Retriever"]["precision"] == 1.0
    assert metrics["Retriever"]["ndcg"] == 1.0

    metrics = eval_result.calculate_metrics(document_scope="answer")

    assert metrics["Retriever"]["mrr"] == 1.0
    assert metrics["Retriever"]["map"] == 1.0
    assert metrics["Retriever"]["recall_multi_hit"] == 1.0
    assert metrics["Retriever"]["recall_single_hit"] == 1.0
    assert metrics["Retriever"]["precision"] == 0.2
    assert metrics["Retriever"]["ndcg"] == 1.0

    metrics = eval_result.calculate_metrics(document_scope="document_id_or_answer")

    assert metrics["Retriever"]["mrr"] == 1.0
    assert metrics["Retriever"]["map"] == 1.0
    assert metrics["Retriever"]["recall_multi_hit"] == 1.0
    assert metrics["Retriever"]["recall_single_hit"] == 1.0
    assert metrics["Retriever"]["precision"] == 0.2
    assert metrics["Retriever"]["ndcg"] == 1.0


@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
@pytest.mark.parametrize("document_store_with_docs", ["memory"], indirect=True)
def test_document_search_eval_document_scope(retriever_with_docs):
    pipeline = DocumentSearchPipeline(retriever=retriever_with_docs)
    eval_result: EvaluationResult = pipeline.eval(
        labels=DOC_SEARCH_EVAL_LABELS,
        params={"Retriever": {"top_k": 5}},
        context_matching_min_length=20,  # artificially set down min_length to see if context matching is working properly
    )

    metrics = eval_result.calculate_metrics(document_scope="document_id")

    assert metrics["Retriever"]["mrr"] == 0.5
    assert metrics["Retriever"]["map"] == 0.5
    assert metrics["Retriever"]["recall_multi_hit"] == 0.5
    assert metrics["Retriever"]["recall_single_hit"] == 0.5
    assert metrics["Retriever"]["precision"] == 0.1
    assert metrics["Retriever"]["ndcg"] == 0.5

    metrics = eval_result.calculate_metrics(document_scope="context")

    assert metrics["Retriever"]["mrr"] == 1.0
    assert metrics["Retriever"]["map"] == 1.0
    assert metrics["Retriever"]["recall_multi_hit"] == 1.0
    assert metrics["Retriever"]["recall_single_hit"] == 1.0
    assert metrics["Retriever"]["precision"] == 1.0
    assert metrics["Retriever"]["ndcg"] == 1.0

    metrics = eval_result.calculate_metrics(document_scope="document_id_and_context")

    assert metrics["Retriever"]["mrr"] == 0.5
    assert metrics["Retriever"]["map"] == 0.5
    assert metrics["Retriever"]["recall_multi_hit"] == 0.5
    assert metrics["Retriever"]["recall_single_hit"] == 0.5
    assert metrics["Retriever"]["precision"] == 0.1
    assert metrics["Retriever"]["ndcg"] == 0.5

    metrics = eval_result.calculate_metrics(document_scope="document_id_or_context")

    assert metrics["Retriever"]["mrr"] == 1.0
    assert metrics["Retriever"]["map"] == 1.0
    assert metrics["Retriever"]["recall_multi_hit"] == 1.0
    assert metrics["Retriever"]["recall_single_hit"] == 1.0
    assert metrics["Retriever"]["precision"] == 1.0
    assert metrics["Retriever"]["ndcg"] == 1.0

    metrics = eval_result.calculate_metrics(document_scope="answer")

    assert metrics["Retriever"]["mrr"] == 0.0
    assert metrics["Retriever"]["map"] == 0.0
    assert metrics["Retriever"]["recall_multi_hit"] == 0.0
    assert metrics["Retriever"]["recall_single_hit"] == 0.0
    assert metrics["Retriever"]["precision"] == 0.0
    assert metrics["Retriever"]["ndcg"] == 0.0

    metrics = eval_result.calculate_metrics(document_scope="document_id_or_answer")

    assert metrics["Retriever"]["mrr"] == 0.5
    assert metrics["Retriever"]["map"] == 0.5
    assert metrics["Retriever"]["recall_multi_hit"] == 0.5
    assert metrics["Retriever"]["recall_single_hit"] == 0.5
    assert metrics["Retriever"]["precision"] == 0.1
    assert metrics["Retriever"]["ndcg"] == 0.5


@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
@pytest.mark.parametrize("document_store_with_docs", ["memory"], indirect=True)
def test_document_search_id_only_eval_document_scope(retriever_with_docs):
    pipeline = DocumentSearchPipeline(retriever=retriever_with_docs)
    eval_result: EvaluationResult = pipeline.eval(
        labels=DOC_SEARCH_ID_EVAL_LABELS,
        params={"Retriever": {"top_k": 5}},
        context_matching_min_length=20,  # artificially set down min_length to see if context matching is working properly
    )

    metrics = eval_result.calculate_metrics(document_scope="document_id")

    assert metrics["Retriever"]["mrr"] == 0.5
    assert metrics["Retriever"]["map"] == 0.5
    assert metrics["Retriever"]["recall_multi_hit"] == 0.5
    assert metrics["Retriever"]["recall_single_hit"] == 0.5
    assert metrics["Retriever"]["precision"] == 0.1
    assert metrics["Retriever"]["ndcg"] == 0.5

    metrics = eval_result.calculate_metrics(document_scope="context")

    assert metrics["Retriever"]["mrr"] == 0.0
    assert metrics["Retriever"]["map"] == 0.0
    assert metrics["Retriever"]["recall_multi_hit"] == 0.0
    assert metrics["Retriever"]["recall_single_hit"] == 0.0
    assert metrics["Retriever"]["precision"] == 0.0
    assert metrics["Retriever"]["ndcg"] == 0.0

    metrics = eval_result.calculate_metrics(document_scope="document_id_and_context")

    assert metrics["Retriever"]["mrr"] == 0.0
    assert metrics["Retriever"]["map"] == 0.0
    assert metrics["Retriever"]["recall_multi_hit"] == 0.0
    assert metrics["Retriever"]["recall_single_hit"] == 0.0
    assert metrics["Retriever"]["precision"] == 0.0
    assert metrics["Retriever"]["ndcg"] == 0.0

    metrics = eval_result.calculate_metrics(document_scope="document_id_or_context")

    assert metrics["Retriever"]["mrr"] == 0.5
    assert metrics["Retriever"]["map"] == 0.5
    assert metrics["Retriever"]["recall_multi_hit"] == 0.5
    assert metrics["Retriever"]["recall_single_hit"] == 0.5
    assert metrics["Retriever"]["precision"] == 0.1
    assert metrics["Retriever"]["ndcg"] == 0.5

    metrics = eval_result.calculate_metrics(document_scope="answer")

    assert metrics["Retriever"]["mrr"] == 0.0
    assert metrics["Retriever"]["map"] == 0.0
    assert metrics["Retriever"]["recall_multi_hit"] == 0.0
    assert metrics["Retriever"]["recall_single_hit"] == 0.0
    assert metrics["Retriever"]["precision"] == 0.0
    assert metrics["Retriever"]["ndcg"] == 0.0

    metrics = eval_result.calculate_metrics(document_scope="document_id_or_answer")

    assert metrics["Retriever"]["mrr"] == 0.5
    assert metrics["Retriever"]["map"] == 0.5
    assert metrics["Retriever"]["recall_multi_hit"] == 0.5
    assert metrics["Retriever"]["recall_single_hit"] == 0.5
    assert metrics["Retriever"]["precision"] == 0.1
    assert metrics["Retriever"]["ndcg"] == 0.5


@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
@pytest.mark.parametrize("document_store_with_docs", ["memory"], indirect=True)
def test_file_search_eval_document_scope(retriever_with_docs):
    pipeline = DocumentSearchPipeline(retriever=retriever_with_docs)
    eval_result: EvaluationResult = pipeline.eval(
        labels=FILE_SEARCH_EVAL_LABELS,
        params={"Retriever": {"top_k": 5}},
        context_matching_min_length=20,  # artificially set down min_length to see if context matching is working properly
        custom_document_id_field="name",
    )

    metrics = eval_result.calculate_metrics(document_scope="document_id")

    assert metrics["Retriever"]["mrr"] == 0.6
    assert metrics["Retriever"]["map"] == 0.6
    assert metrics["Retriever"]["recall_multi_hit"] == 1.0
    assert metrics["Retriever"]["recall_single_hit"] == 1.0
    assert metrics["Retriever"]["precision"] == 0.2
    assert metrics["Retriever"]["ndcg"] == pytest.approx(0.6934, 0.0001)

    metrics = eval_result.calculate_metrics(document_scope="context")

    assert metrics["Retriever"]["mrr"] == 0.0
    assert metrics["Retriever"]["map"] == 0.0
    assert metrics["Retriever"]["recall_multi_hit"] == 0.0
    assert metrics["Retriever"]["recall_single_hit"] == 0.0
    assert metrics["Retriever"]["precision"] == 0.0
    assert metrics["Retriever"]["ndcg"] == 0.0

    metrics = eval_result.calculate_metrics(document_scope="document_id_and_context")

    assert metrics["Retriever"]["mrr"] == 0.0
    assert metrics["Retriever"]["map"] == 0.0
    assert metrics["Retriever"]["recall_multi_hit"] == 0.0
    assert metrics["Retriever"]["recall_single_hit"] == 0.0
    assert metrics["Retriever"]["precision"] == 0.0
    assert metrics["Retriever"]["ndcg"] == 0.0

    metrics = eval_result.calculate_metrics(document_scope="document_id_or_context")

    assert metrics["Retriever"]["mrr"] == 0.6
    assert metrics["Retriever"]["map"] == 0.6
    assert metrics["Retriever"]["recall_multi_hit"] == 1.0
    assert metrics["Retriever"]["recall_single_hit"] == 1.0
    assert metrics["Retriever"]["precision"] == 0.2
    assert metrics["Retriever"]["ndcg"] == pytest.approx(0.6934, 0.0001)

    metrics = eval_result.calculate_metrics(document_scope="answer")

    assert metrics["Retriever"]["mrr"] == 0.0
    assert metrics["Retriever"]["map"] == 0.0
    assert metrics["Retriever"]["recall_multi_hit"] == 0.0
    assert metrics["Retriever"]["recall_single_hit"] == 0.0
    assert metrics["Retriever"]["precision"] == 0.0
    assert metrics["Retriever"]["ndcg"] == 0.0

    metrics = eval_result.calculate_metrics(document_scope="document_id_or_answer")

    assert metrics["Retriever"]["mrr"] == 0.6
    assert metrics["Retriever"]["map"] == 0.6
    assert metrics["Retriever"]["recall_multi_hit"] == 1.0
    assert metrics["Retriever"]["recall_single_hit"] == 1.0
    assert metrics["Retriever"]["precision"] == 0.2
    assert metrics["Retriever"]["ndcg"] == pytest.approx(0.6934, 0.0001)


@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
@pytest.mark.parametrize("document_store_with_docs", ["memory"], indirect=True)
@pytest.mark.parametrize(
    "document_scope",
    ["document_id", "context", "document_id_and_context", "document_id_or_context", "answer", "document_id_or_answer"],
)
def test_extractive_qa_eval_document_scope_no_answer(retriever_with_docs, document_scope):
    pipeline = DocumentSearchPipeline(retriever=retriever_with_docs)
    eval_result: EvaluationResult = pipeline.eval(
        labels=NO_ANSWER_EVAL_LABELS,
        params={"Retriever": {"top_k": 5}},
        context_matching_min_length=20,  # artificially set down min_length to see if context matching is working properly
    )

    metrics = eval_result.calculate_metrics(document_scope=document_scope)

    assert metrics["Retriever"]["mrr"] == 1.0
    assert metrics["Retriever"]["map"] == 1.0
    assert metrics["Retriever"]["recall_multi_hit"] == 1.0
    assert metrics["Retriever"]["recall_single_hit"] == 1.0
    assert metrics["Retriever"]["precision"] == 1.0
    assert metrics["Retriever"]["ndcg"] == 1.0


@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
@pytest.mark.parametrize("document_store_with_docs", ["memory"], indirect=True)
@pytest.mark.parametrize("reader", ["farm"], indirect=True)
def test_extractive_qa_eval_answer_scope(reader, retriever_with_docs, eval_labels):
    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever_with_docs)
    eval_result: EvaluationResult = pipeline.eval(
        labels=eval_labels,
        params={"Retriever": {"top_k": 5}},
        sas_model_name_or_path="sentence-transformers/paraphrase-MiniLM-L3-v2",
        context_matching_min_length=20,  # artificially set down min_length to see if context matching is working properly
    )

    metrics = eval_result.calculate_metrics(answer_scope="any")

    assert metrics["Retriever"]["mrr"] == 1.0
    assert metrics["Retriever"]["map"] == 1.0
    assert metrics["Retriever"]["recall_multi_hit"] == 1.0
    assert metrics["Retriever"]["recall_single_hit"] == 1.0
    assert metrics["Retriever"]["precision"] == 0.2
    assert metrics["Retriever"]["ndcg"] == 1.0
    assert metrics["Reader"]["exact_match"] == 1.0
    assert metrics["Reader"]["f1"] == 1.0
    assert metrics["Reader"]["sas"] == pytest.approx(1.0)

    metrics = eval_result.calculate_metrics(answer_scope="context")

    assert metrics["Retriever"]["mrr"] == 1.0
    assert metrics["Retriever"]["map"] == 1.0
    assert metrics["Retriever"]["recall_multi_hit"] == 1.0
    assert metrics["Retriever"]["recall_single_hit"] == 1.0
    assert metrics["Retriever"]["precision"] == 0.2
    assert metrics["Retriever"]["ndcg"] == 1.0
    assert metrics["Reader"]["exact_match"] == 1.0
    assert metrics["Reader"]["f1"] == 1.0
    assert metrics["Reader"]["sas"] == pytest.approx(1.0)

    metrics = eval_result.calculate_metrics(answer_scope="document_id")

    assert metrics["Retriever"]["mrr"] == 0.5
    assert metrics["Retriever"]["map"] == 0.5
    assert metrics["Retriever"]["recall_multi_hit"] == 0.5
    assert metrics["Retriever"]["recall_single_hit"] == 0.5
    assert metrics["Retriever"]["precision"] == 0.1
    assert metrics["Retriever"]["ndcg"] == 0.5
    assert metrics["Reader"]["exact_match"] == 0.5
    assert metrics["Reader"]["f1"] == 0.5
    assert metrics["Reader"]["sas"] == pytest.approx(0.5)

    metrics = eval_result.calculate_metrics(answer_scope="document_id_and_context")

    assert metrics["Retriever"]["mrr"] == 0.5
    assert metrics["Retriever"]["map"] == 0.5
    assert metrics["Retriever"]["recall_multi_hit"] == 0.5
    assert metrics["Retriever"]["recall_single_hit"] == 0.5
    assert metrics["Retriever"]["precision"] == 0.1
    assert metrics["Retriever"]["ndcg"] == 0.5
    assert metrics["Reader"]["exact_match"] == 0.5
    assert metrics["Reader"]["f1"] == 0.5
    assert metrics["Reader"]["sas"] == pytest.approx(0.5)


@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
@pytest.mark.parametrize("document_store_with_docs", ["memory"], indirect=True)
@pytest.mark.parametrize("reader", ["farm"], indirect=True)
def test_extractive_qa_eval_answer_document_scope_combinations(reader, retriever_with_docs, caplog, eval_labels):
    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever_with_docs)
    eval_result: EvaluationResult = pipeline.eval(
        labels=eval_labels,
        params={"Retriever": {"top_k": 5}},
        sas_model_name_or_path="sentence-transformers/paraphrase-MiniLM-L3-v2",
        context_matching_min_length=20,  # artificially set down min_length to see if context matching is working properly
    )

    # valid values for non default answer_scopes
    with caplog.at_level(logging.WARNING):
        metrics = eval_result.calculate_metrics(document_scope="document_id_or_answer", answer_scope="context")
        metrics = eval_result.calculate_metrics(document_scope="answer", answer_scope="context")
        assert "You specified a non-answer document_scope together with a non-default answer_scope" not in caplog.text

    with caplog.at_level(logging.WARNING):
        metrics = eval_result.calculate_metrics(document_scope="document_id", answer_scope="context")
        assert "You specified a non-answer document_scope together with a non-default answer_scope" in caplog.text

    with caplog.at_level(logging.WARNING):
        metrics = eval_result.calculate_metrics(document_scope="context", answer_scope="context")
        assert "You specified a non-answer document_scope together with a non-default answer_scope" in caplog.text

    with caplog.at_level(logging.WARNING):
        metrics = eval_result.calculate_metrics(document_scope="document_id_and_context", answer_scope="context")
        assert "You specified a non-answer document_scope together with a non-default answer_scope" in caplog.text

    with caplog.at_level(logging.WARNING):
        metrics = eval_result.calculate_metrics(document_scope="document_id_or_context", answer_scope="context")
        assert "You specified a non-answer document_scope together with a non-default answer_scope" in caplog.text


@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
@pytest.mark.parametrize("document_store_with_docs", ["memory"], indirect=True)
@pytest.mark.parametrize("reader", ["farm"], indirect=True)
def test_extractive_qa_eval_simulated_top_k_reader(reader, retriever_with_docs, eval_labels):
    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever_with_docs)
    eval_result: EvaluationResult = pipeline.eval(
        labels=eval_labels,
        params={"Retriever": {"top_k": 5}},
        sas_model_name_or_path="sentence-transformers/paraphrase-MiniLM-L3-v2",
    )

    metrics_top_1 = eval_result.calculate_metrics(simulated_top_k_reader=1, document_scope="document_id")

    assert metrics_top_1["Reader"]["exact_match"] == 0.5
    assert metrics_top_1["Reader"]["f1"] == 0.5
    assert metrics_top_1["Reader"]["sas"] == pytest.approx(0.6003, abs=1e-4)
    assert metrics_top_1["Retriever"]["mrr"] == 0.5
    assert metrics_top_1["Retriever"]["map"] == 0.5
    assert metrics_top_1["Retriever"]["recall_multi_hit"] == 0.5
    assert metrics_top_1["Retriever"]["recall_single_hit"] == 0.5
    assert metrics_top_1["Retriever"]["precision"] == 0.1
    assert metrics_top_1["Retriever"]["ndcg"] == 0.5

    metrics_top_2 = eval_result.calculate_metrics(simulated_top_k_reader=2, document_scope="document_id")

    assert metrics_top_2["Reader"]["exact_match"] == 0.5
    assert metrics_top_2["Reader"]["f1"] == 0.5
    assert metrics_top_2["Reader"]["sas"] == pytest.approx(0.6003, abs=1e-4)
    assert metrics_top_2["Retriever"]["mrr"] == 0.5
    assert metrics_top_2["Retriever"]["map"] == 0.5
    assert metrics_top_2["Retriever"]["recall_multi_hit"] == 0.5
    assert metrics_top_2["Retriever"]["recall_single_hit"] == 0.5
    assert metrics_top_2["Retriever"]["precision"] == 0.1
    assert metrics_top_2["Retriever"]["ndcg"] == 0.5

    metrics_top_5 = eval_result.calculate_metrics(simulated_top_k_reader=5, document_scope="document_id")

    assert metrics_top_5["Reader"]["exact_match"] == 1.0
    assert metrics_top_5["Reader"]["f1"] == 1.0
    assert metrics_top_5["Reader"]["sas"] == pytest.approx(1.0, abs=1e-4)
    assert metrics_top_5["Retriever"]["mrr"] == 0.5
    assert metrics_top_5["Retriever"]["map"] == 0.5
    assert metrics_top_5["Retriever"]["recall_multi_hit"] == 0.5
    assert metrics_top_5["Retriever"]["recall_single_hit"] == 0.5
    assert metrics_top_5["Retriever"]["precision"] == 0.1
    assert metrics_top_5["Retriever"]["ndcg"] == 0.5


@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
@pytest.mark.parametrize("document_store_with_docs", ["memory"], indirect=True)
@pytest.mark.parametrize("reader", ["farm"], indirect=True)
def test_extractive_qa_eval_simulated_top_k_retriever(reader, retriever_with_docs, eval_labels):
    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever_with_docs)
    eval_result: EvaluationResult = pipeline.eval(labels=eval_labels, params={"Retriever": {"top_k": 5}})

    metrics_top_10 = eval_result.calculate_metrics(document_scope="document_id")

    assert metrics_top_10["Reader"]["exact_match"] == 1.0
    assert metrics_top_10["Reader"]["f1"] == 1.0
    assert metrics_top_10["Retriever"]["mrr"] == 0.5
    assert metrics_top_10["Retriever"]["map"] == 0.5
    assert metrics_top_10["Retriever"]["recall_multi_hit"] == 0.5
    assert metrics_top_10["Retriever"]["recall_single_hit"] == 0.5
    assert metrics_top_10["Retriever"]["precision"] == 0.1
    assert metrics_top_10["Retriever"]["ndcg"] == 0.5

    metrics_top_1 = eval_result.calculate_metrics(simulated_top_k_retriever=1, document_scope="document_id")

    assert metrics_top_1["Reader"]["exact_match"] == 1.0
    assert metrics_top_1["Reader"]["f1"] == 1.0
    assert metrics_top_1["Retriever"]["mrr"] == 0.5
    assert metrics_top_1["Retriever"]["map"] == 0.5
    assert metrics_top_1["Retriever"]["recall_multi_hit"] == 0.5
    assert metrics_top_1["Retriever"]["recall_single_hit"] == 0.5
    assert metrics_top_1["Retriever"]["precision"] == 0.5
    assert metrics_top_1["Retriever"]["ndcg"] == 0.5

    metrics_top_2 = eval_result.calculate_metrics(simulated_top_k_retriever=2, document_scope="document_id")

    assert metrics_top_2["Reader"]["exact_match"] == 1.0
    assert metrics_top_2["Reader"]["f1"] == 1.0
    assert metrics_top_2["Retriever"]["mrr"] == 0.5
    assert metrics_top_2["Retriever"]["map"] == 0.5
    assert metrics_top_2["Retriever"]["recall_multi_hit"] == 0.5
    assert metrics_top_2["Retriever"]["recall_single_hit"] == 0.5
    assert metrics_top_2["Retriever"]["precision"] == 0.25
    assert metrics_top_2["Retriever"]["ndcg"] == 0.5

    metrics_top_3 = eval_result.calculate_metrics(simulated_top_k_retriever=3, document_scope="document_id")

    assert metrics_top_3["Reader"]["exact_match"] == 1.0
    assert metrics_top_3["Reader"]["f1"] == 1.0
    assert metrics_top_3["Retriever"]["mrr"] == 0.5
    assert metrics_top_3["Retriever"]["map"] == 0.5
    assert metrics_top_3["Retriever"]["recall_multi_hit"] == 0.5
    assert metrics_top_3["Retriever"]["recall_single_hit"] == 0.5
    assert metrics_top_3["Retriever"]["precision"] == 1.0 / 6
    assert metrics_top_3["Retriever"]["ndcg"] == 0.5


@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
@pytest.mark.parametrize("document_store_with_docs", ["memory"], indirect=True)
@pytest.mark.parametrize("reader", ["farm"], indirect=True)
def test_extractive_qa_eval_simulated_top_k_reader_and_retriever(reader, retriever_with_docs, eval_labels):
    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever_with_docs)
    eval_result: EvaluationResult = pipeline.eval(labels=eval_labels, params={"Retriever": {"top_k": 10}})

    metrics_top_10 = eval_result.calculate_metrics(simulated_top_k_reader=1, document_scope="document_id")

    assert metrics_top_10["Reader"]["exact_match"] == 0.5
    assert metrics_top_10["Reader"]["f1"] == 0.5
    assert metrics_top_10["Retriever"]["mrr"] == 0.5
    assert metrics_top_10["Retriever"]["map"] == 0.5
    assert metrics_top_10["Retriever"]["recall_multi_hit"] == 0.5
    assert metrics_top_10["Retriever"]["recall_single_hit"] == 0.5
    assert metrics_top_10["Retriever"]["precision"] == 0.1
    assert metrics_top_10["Retriever"]["ndcg"] == 0.5

    metrics_top_1 = eval_result.calculate_metrics(
        simulated_top_k_reader=1, simulated_top_k_retriever=1, document_scope="document_id"
    )

    assert metrics_top_1["Reader"]["exact_match"] == 1.0
    assert metrics_top_1["Reader"]["f1"] == 1.0

    assert metrics_top_1["Retriever"]["mrr"] == 0.5
    assert metrics_top_1["Retriever"]["map"] == 0.5
    assert metrics_top_1["Retriever"]["recall_multi_hit"] == 0.5
    assert metrics_top_1["Retriever"]["recall_single_hit"] == 0.5
    assert metrics_top_1["Retriever"]["precision"] == 0.5
    assert metrics_top_1["Retriever"]["ndcg"] == 0.5

    metrics_top_2 = eval_result.calculate_metrics(
        simulated_top_k_reader=1, simulated_top_k_retriever=2, document_scope="document_id"
    )

    assert metrics_top_2["Reader"]["exact_match"] == 0.5
    assert metrics_top_2["Reader"]["f1"] == 0.5
    assert metrics_top_2["Retriever"]["mrr"] == 0.5
    assert metrics_top_2["Retriever"]["map"] == 0.5
    assert metrics_top_2["Retriever"]["recall_multi_hit"] == 0.5
    assert metrics_top_2["Retriever"]["recall_single_hit"] == 0.5
    assert metrics_top_2["Retriever"]["precision"] == 0.25
    assert metrics_top_2["Retriever"]["ndcg"] == 0.5

    metrics_top_3 = eval_result.calculate_metrics(
        simulated_top_k_reader=1, simulated_top_k_retriever=3, document_scope="document_id"
    )

    assert metrics_top_3["Reader"]["exact_match"] == 0.5
    assert metrics_top_3["Reader"]["f1"] == 0.5
    assert metrics_top_3["Retriever"]["mrr"] == 0.5
    assert metrics_top_3["Retriever"]["map"] == 0.5
    assert metrics_top_3["Retriever"]["recall_multi_hit"] == 0.5
    assert metrics_top_3["Retriever"]["recall_single_hit"] == 0.5
    assert metrics_top_3["Retriever"]["precision"] == 1.0 / 6
    assert metrics_top_3["Retriever"]["ndcg"] == 0.5


@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
@pytest.mark.parametrize("document_store_with_docs", ["memory"], indirect=True)
@pytest.mark.parametrize("reader", ["farm"], indirect=True)
def test_extractive_qa_eval_isolated(reader, retriever_with_docs, eval_labels):
    labels = deepcopy(eval_labels)
    # Copy one of the labels and change only the answer have a label with a different answer but same Document
    label_copy = deepcopy(labels[0].labels[0])
    label_copy.answer = Answer(answer="I", offsets_in_context=[Span(21, 22)])
    labels[0].labels.append(label_copy)
    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever_with_docs)
    eval_result: EvaluationResult = pipeline.eval(
        labels=labels,
        sas_model_name_or_path="sentence-transformers/paraphrase-MiniLM-L3-v2",
        add_isolated_node_eval=True,
    )

    metrics_top_1 = eval_result.calculate_metrics(simulated_top_k_reader=1, document_scope="document_id")

    assert metrics_top_1["Reader"]["exact_match"] == 0.5
    assert metrics_top_1["Reader"]["f1"] == 0.5
    assert metrics_top_1["Reader"]["sas"] == pytest.approx(0.6003, abs=1e-4)
    assert metrics_top_1["Retriever"]["mrr"] == 0.5
    assert metrics_top_1["Retriever"]["map"] == 0.5
    assert metrics_top_1["Retriever"]["recall_multi_hit"] == 0.5
    assert metrics_top_1["Retriever"]["recall_single_hit"] == 0.5
    assert metrics_top_1["Retriever"]["precision"] == 1.0 / 10
    assert metrics_top_1["Retriever"]["ndcg"] == 0.5

    metrics_top_1 = eval_result.calculate_metrics(simulated_top_k_reader=1, eval_mode="isolated")

    assert metrics_top_1["Reader"]["exact_match"] == 1.0
    assert metrics_top_1["Reader"]["f1"] == 1.0
    assert metrics_top_1["Reader"]["sas"] == pytest.approx(1.0, abs=1e-4)

    # Check if same Document in MultiLabel got deduplicated
    assert labels[0].labels[0].id == labels[0].labels[1].id
    reader_eval_df = eval_result.node_results["Reader"]
    isolated_reader_eval_df = reader_eval_df[reader_eval_df["eval_mode"] == "isolated"]
    assert len(isolated_reader_eval_df) == len(labels) * reader.top_k_per_candidate


@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
@pytest.mark.parametrize("document_store_with_docs", ["memory"], indirect=True)
@pytest.mark.parametrize("reader", ["farm"], indirect=True)
def test_extractive_qa_eval_wrong_examples(reader, retriever_with_docs):
    labels = [
        MultiLabel(
            labels=[
                Label(
                    query="Who lives in Berlin?",
                    answer=Answer(answer="Carla", offsets_in_context=[Span(11, 16)]),
                    document=Document(
                        id="a0747b83aea0b60c4b114b15476dd32d",
                        content_type="text",
                        content="My name is Carla and I live in Berlin",
                    ),
                    is_correct_answer=True,
                    is_correct_document=True,
                    origin="gold-label",
                )
            ]
        ),
        MultiLabel(
            labels=[
                Label(
                    query="Who lives in Munich?",
                    answer=Answer(answer="Pete", offsets_in_context=[Span(11, 16)]),
                    document=Document(
                        id="something_else", content_type="text", content="My name is Pete and I live in Munich"
                    ),
                    is_correct_answer=True,
                    is_correct_document=True,
                    origin="gold-label",
                )
            ]
        ),
    ]

    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever_with_docs)
    eval_result: EvaluationResult = pipeline.eval(labels=labels, params={"Retriever": {"top_k": 5}})

    wrongs_retriever = eval_result.wrong_examples(node="Retriever", n=1)
    wrongs_reader = eval_result.wrong_examples(node="Reader", n=1)

    assert len(wrongs_retriever) == 1
    assert len(wrongs_reader) == 1


@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
@pytest.mark.parametrize("document_store_with_docs", ["memory"], indirect=True)
@pytest.mark.parametrize("reader", ["farm"], indirect=True)
def test_extractive_qa_print_eval_report(reader, retriever_with_docs):
    labels = [
        MultiLabel(
            labels=[
                Label(
                    query="Who lives in Berlin?",
                    answer=Answer(answer="Carla", offsets_in_context=[Span(11, 16)]),
                    document=Document(
                        id="a0747b83aea0b60c4b114b15476dd32d",
                        content_type="text",
                        content="My name is Carla and I live in Berlin",
                    ),
                    is_correct_answer=True,
                    is_correct_document=True,
                    origin="gold-label",
                )
            ]
        ),
        MultiLabel(
            labels=[
                Label(
                    query="Who lives in Munich?",
                    answer=Answer(answer="Pete", offsets_in_context=[Span(11, 16)]),
                    document=Document(
                        id="something_else", content_type="text", content="My name is Pete and I live in Munich"
                    ),
                    is_correct_answer=True,
                    is_correct_document=True,
                    origin="gold-label",
                )
            ]
        ),
    ]

    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever_with_docs)
    eval_result: EvaluationResult = pipeline.eval(labels=labels, params={"Retriever": {"top_k": 5}})
    pipeline.print_eval_report(eval_result)

    # in addition with labels as input to reader node rather than output of retriever node
    eval_result: EvaluationResult = pipeline.eval(
        labels=labels, params={"Retriever": {"top_k": 5}}, add_isolated_node_eval=True
    )
    pipeline.print_eval_report(eval_result)


@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
@pytest.mark.parametrize("document_store_with_docs", ["memory"], indirect=True)
def test_document_search_calculate_metrics(retriever_with_docs, eval_labels):
    pipeline = DocumentSearchPipeline(retriever=retriever_with_docs)
    eval_result: EvaluationResult = pipeline.eval(labels=eval_labels, params={"Retriever": {"top_k": 5}})

    metrics = eval_result.calculate_metrics(document_scope="document_id")

    assert "Retriever" in eval_result
    assert len(eval_result) == 1
    retriever_result = eval_result["Retriever"]
    retriever_berlin = retriever_result[retriever_result["query"] == "Who lives in Berlin?"]
    retriever_munich = retriever_result[retriever_result["query"] == "Who lives in Munich?"]

    assert (
        retriever_berlin[retriever_berlin["rank"] == 1]["document_id"].iloc[0]
        in retriever_berlin[retriever_berlin["rank"] == 1]["gold_document_ids"].iloc[0]
    )
    assert (
        retriever_munich[retriever_munich["rank"] == 1]["document_id"].iloc[0]
        not in retriever_munich[retriever_munich["rank"] == 1]["gold_document_ids"].iloc[0]
    )
    assert metrics["Retriever"]["mrr"] == 0.5
    assert metrics["Retriever"]["map"] == 0.5
    assert metrics["Retriever"]["recall_multi_hit"] == 0.5
    assert metrics["Retriever"]["recall_single_hit"] == 0.5
    assert metrics["Retriever"]["precision"] == 0.1
    assert metrics["Retriever"]["ndcg"] == 0.5


@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
@pytest.mark.parametrize("document_store_with_docs", ["memory"], indirect=True)
def test_document_search_isolated(retriever_with_docs, eval_labels):
    pipeline = DocumentSearchPipeline(retriever=retriever_with_docs)
    # eval run must not fail even though no node supports add_isolated_node_eval
    eval_result: EvaluationResult = pipeline.eval(
        labels=eval_labels, params={"Retriever": {"top_k": 5}}, add_isolated_node_eval=True
    )

    metrics = eval_result.calculate_metrics(document_scope="document_id")

    assert "Retriever" in eval_result
    assert len(eval_result) == 1
    retriever_result = eval_result["Retriever"]
    retriever_berlin = retriever_result[retriever_result["query"] == "Who lives in Berlin?"]
    retriever_munich = retriever_result[retriever_result["query"] == "Who lives in Munich?"]

    assert (
        retriever_berlin[retriever_berlin["rank"] == 1]["document_id"].iloc[0]
        in retriever_berlin[retriever_berlin["rank"] == 1]["gold_document_ids"].iloc[0]
    )
    assert (
        retriever_munich[retriever_munich["rank"] == 1]["document_id"].iloc[0]
        not in retriever_munich[retriever_munich["rank"] == 1]["gold_document_ids"].iloc[0]
    )
    assert metrics["Retriever"]["mrr"] == 0.5
    assert metrics["Retriever"]["map"] == 0.5
    assert metrics["Retriever"]["recall_multi_hit"] == 0.5
    assert metrics["Retriever"]["recall_single_hit"] == 0.5
    assert metrics["Retriever"]["precision"] == 0.1
    assert metrics["Retriever"]["ndcg"] == 0.5

    isolated_metrics = eval_result.calculate_metrics(document_scope="document_id", eval_mode="isolated")
    # empty metrics for nodes that do not support add_isolated_node_eval
    assert isolated_metrics["Retriever"] == {}


@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
@pytest.mark.parametrize("document_store_with_docs", ["memory"], indirect=True)
def test_faq_calculate_metrics(retriever_with_docs, eval_labels):
    pipeline = FAQPipeline(retriever=retriever_with_docs)
    eval_result: EvaluationResult = pipeline.eval(labels=eval_labels, params={"Retriever": {"top_k": 5}})

    metrics = eval_result.calculate_metrics(document_scope="document_id")

    assert "Retriever" in eval_result
    assert "Docs2Answers" in eval_result
    assert len(eval_result) == 2

    assert metrics["Retriever"]["mrr"] == 0.5
    assert metrics["Retriever"]["map"] == 0.5
    assert metrics["Retriever"]["recall_multi_hit"] == 0.5
    assert metrics["Retriever"]["recall_single_hit"] == 0.5
    assert metrics["Retriever"]["precision"] == 0.1
    assert metrics["Retriever"]["ndcg"] == 0.5
    assert metrics["Docs2Answers"]["exact_match"] == 0.0
    assert metrics["Docs2Answers"]["f1"] == 0.0


@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
@pytest.mark.parametrize("document_store_with_docs", ["memory"], indirect=True)
@pytest.mark.parametrize("reader", ["farm"], indirect=True)
def test_extractive_qa_eval_translation(reader, retriever_with_docs, eval_labels):
    # FIXME it makes no sense to have DE->EN input and DE->EN output, right?
    #  Yet switching direction breaks the test. TO BE FIXED.
    input_translator = TransformersTranslator(model_name_or_path="Helsinki-NLP/opus-mt-de-en")
    output_translator = TransformersTranslator(model_name_or_path="Helsinki-NLP/opus-mt-de-en")

    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever_with_docs)
    pipeline = TranslationWrapperPipeline(
        input_translator=input_translator, output_translator=output_translator, pipeline=pipeline
    )
    eval_result: EvaluationResult = pipeline.eval(labels=eval_labels, params={"Retriever": {"top_k": 5}})

    metrics = eval_result.calculate_metrics(document_scope="document_id")

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
    assert metrics["Retriever"]["precision"] == 0.1
    assert metrics["Retriever"]["ndcg"] == 0.5

    assert metrics["OutputTranslator"]["exact_match"] == 1.0
    assert metrics["OutputTranslator"]["f1"] == 1.0
    assert metrics["OutputTranslator"]["mrr"] == 0.5
    assert metrics["OutputTranslator"]["map"] == 0.5
    assert metrics["OutputTranslator"]["recall_multi_hit"] == 0.5
    assert metrics["OutputTranslator"]["recall_single_hit"] == 0.5
    assert metrics["OutputTranslator"]["precision"] == 0.1
    assert metrics["OutputTranslator"]["ndcg"] == 0.5


@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
@pytest.mark.parametrize("document_store_with_docs", ["memory"], indirect=True)
def test_question_generation_eval(retriever_with_docs, question_generator, eval_labels):
    pipeline = RetrieverQuestionGenerationPipeline(retriever=retriever_with_docs, question_generator=question_generator)

    eval_result: EvaluationResult = pipeline.eval(labels=eval_labels, params={"Retriever": {"top_k": 5}})

    metrics = eval_result.calculate_metrics(document_scope="document_id")

    assert "Retriever" in eval_result
    assert "QuestionGenerator" in eval_result
    assert len(eval_result) == 2

    assert metrics["Retriever"]["mrr"] == 0.5
    assert metrics["Retriever"]["map"] == 0.5
    assert metrics["Retriever"]["recall_multi_hit"] == 0.5
    assert metrics["Retriever"]["recall_single_hit"] == 0.5
    assert metrics["Retriever"]["precision"] == 0.1
    assert metrics["Retriever"]["ndcg"] == 0.5

    assert metrics["QuestionGenerator"]["mrr"] == 0.5
    assert metrics["QuestionGenerator"]["map"] == 0.5
    assert metrics["QuestionGenerator"]["recall_multi_hit"] == 0.5
    assert metrics["QuestionGenerator"]["recall_single_hit"] == 0.5
    assert metrics["QuestionGenerator"]["precision"] == 0.1
    assert metrics["QuestionGenerator"]["ndcg"] == 0.5


@pytest.mark.parametrize("document_store_with_docs", ["elasticsearch"], indirect=True)
@pytest.mark.parametrize("reader", ["farm"], indirect=True)
def test_qa_multi_retriever_pipeline_eval(document_store_with_docs, reader, eval_labels):
    es_retriever = BM25Retriever(document_store=document_store_with_docs)
    dpr_retriever = DensePassageRetriever(document_store_with_docs)
    document_store_with_docs.update_embeddings(retriever=dpr_retriever)

    # QA Pipeline with two retrievers, we always want QA output
    pipeline = Pipeline()
    pipeline.add_node(component=TransformersQueryClassifier(), name="QueryClassifier", inputs=["Query"])
    pipeline.add_node(component=dpr_retriever, name="DPRRetriever", inputs=["QueryClassifier.output_1"])
    pipeline.add_node(component=es_retriever, name="ESRetriever", inputs=["QueryClassifier.output_2"])
    pipeline.add_node(component=reader, name="QAReader", inputs=["ESRetriever", "DPRRetriever"])

    # EVAL_QUERIES: 2 go dpr way
    # in Berlin goes es way
    labels = eval_labels + [
        MultiLabel(
            labels=[
                Label(
                    query="in Berlin",
                    answer=Answer(answer="Carla", offsets_in_context=[Span(11, 16)]),
                    document=Document(
                        id="a0747b83aea0b60c4b114b15476dd32d",
                        content_type="text",
                        content="My name is Carla and I live in Berlin",
                    ),
                    is_correct_answer=True,
                    is_correct_document=True,
                    origin="gold-label",
                )
            ]
        )
    ]

    eval_result: EvaluationResult = pipeline.eval(
        labels=labels, params={"ESRetriever": {"top_k": 5}, "DPRRetriever": {"top_k": 5}}
    )

    metrics = eval_result.calculate_metrics(document_scope="document_id")

    assert "ESRetriever" in eval_result
    assert "DPRRetriever" in eval_result
    assert "QAReader" in eval_result
    assert len(eval_result) == 3

    assert metrics["DPRRetriever"]["mrr"] == 0.5
    assert metrics["DPRRetriever"]["map"] == 0.5
    assert metrics["DPRRetriever"]["recall_multi_hit"] == 0.5
    assert metrics["DPRRetriever"]["recall_single_hit"] == 0.5
    assert metrics["DPRRetriever"]["precision"] == 0.1
    assert metrics["DPRRetriever"]["ndcg"] == 0.5

    assert metrics["ESRetriever"]["mrr"] == 1.0
    assert metrics["ESRetriever"]["map"] == 1.0
    assert metrics["ESRetriever"]["recall_multi_hit"] == 1.0
    assert metrics["ESRetriever"]["recall_single_hit"] == 1.0
    assert metrics["ESRetriever"]["precision"] == 0.2
    assert metrics["ESRetriever"]["ndcg"] == 1.0

    assert metrics["QAReader"]["exact_match"] == 1.0
    assert metrics["QAReader"]["f1"] == 1.0


@pytest.mark.parametrize("document_store_with_docs", ["elasticsearch"], indirect=True)
def test_multi_retriever_pipeline_eval(document_store_with_docs, eval_labels):
    es_retriever = BM25Retriever(document_store=document_store_with_docs)
    dpr_retriever = DensePassageRetriever(document_store_with_docs)
    document_store_with_docs.update_embeddings(retriever=dpr_retriever)

    # QA Pipeline with two retrievers, no QA output
    pipeline = Pipeline()
    pipeline.add_node(component=TransformersQueryClassifier(), name="QueryClassifier", inputs=["Query"])
    pipeline.add_node(component=dpr_retriever, name="DPRRetriever", inputs=["QueryClassifier.output_1"])
    pipeline.add_node(component=es_retriever, name="ESRetriever", inputs=["QueryClassifier.output_2"])

    # EVAL_QUERIES: 2 go dpr way
    # in Berlin goes es way
    labels = eval_labels + [
        MultiLabel(
            labels=[
                Label(
                    query="in Berlin",
                    answer=None,
                    document=Document(
                        id="a0747b83aea0b60c4b114b15476dd32d",
                        content_type="text",
                        content="My name is Carla and I live in Berlin",
                    ),
                    is_correct_answer=True,
                    is_correct_document=True,
                    origin="gold-label",
                )
            ]
        )
    ]

    eval_result: EvaluationResult = pipeline.eval(
        labels=labels, params={"ESRetriever": {"top_k": 5}, "DPRRetriever": {"top_k": 5}}
    )

    metrics = eval_result.calculate_metrics(document_scope="document_id")

    assert "ESRetriever" in eval_result
    assert "DPRRetriever" in eval_result
    assert len(eval_result) == 2

    assert metrics["DPRRetriever"]["mrr"] == 0.5
    assert metrics["DPRRetriever"]["map"] == 0.5
    assert metrics["DPRRetriever"]["recall_multi_hit"] == 0.5
    assert metrics["DPRRetriever"]["recall_single_hit"] == 0.5
    assert metrics["DPRRetriever"]["precision"] == 0.1
    assert metrics["DPRRetriever"]["ndcg"] == 0.5

    assert metrics["ESRetriever"]["mrr"] == 1.0
    assert metrics["ESRetriever"]["map"] == 1.0
    assert metrics["ESRetriever"]["recall_multi_hit"] == 1.0
    assert metrics["ESRetriever"]["recall_single_hit"] == 1.0
    assert metrics["ESRetriever"]["precision"] == 0.2
    assert metrics["ESRetriever"]["ndcg"] == 1.0


@pytest.mark.parametrize("document_store_with_docs", ["elasticsearch"], indirect=True)
@pytest.mark.parametrize("reader", ["farm"], indirect=True)
def test_multi_retriever_pipeline_with_asymmetric_qa_eval(document_store_with_docs, reader, eval_labels):
    es_retriever = BM25Retriever(document_store=document_store_with_docs)
    dpr_retriever = DensePassageRetriever(document_store_with_docs)
    document_store_with_docs.update_embeddings(retriever=dpr_retriever)

    # QA Pipeline with two retrievers, we only get QA output from dpr
    pipeline = Pipeline()
    pipeline.add_node(component=TransformersQueryClassifier(), name="QueryClassifier", inputs=["Query"])
    pipeline.add_node(component=dpr_retriever, name="DPRRetriever", inputs=["QueryClassifier.output_1"])
    pipeline.add_node(component=es_retriever, name="ESRetriever", inputs=["QueryClassifier.output_2"])
    pipeline.add_node(component=reader, name="QAReader", inputs=["DPRRetriever"])

    # EVAL_QUERIES: 2 go dpr way
    # in Berlin goes es way
    labels = eval_labels + [
        MultiLabel(
            labels=[
                Label(
                    query="in Berlin",
                    answer=None,
                    document=Document(
                        id="a0747b83aea0b60c4b114b15476dd32d",
                        content_type="text",
                        content="My name is Carla and I live in Berlin",
                    ),
                    is_correct_answer=True,
                    is_correct_document=True,
                    origin="gold-label",
                )
            ]
        )
    ]

    eval_result: EvaluationResult = pipeline.eval(
        labels=labels, params={"ESRetriever": {"top_k": 5}, "DPRRetriever": {"top_k": 5}}
    )

    metrics = eval_result.calculate_metrics(document_scope="document_id")

    assert "ESRetriever" in eval_result
    assert "DPRRetriever" in eval_result
    assert "QAReader" in eval_result
    assert len(eval_result) == 3

    assert metrics["DPRRetriever"]["mrr"] == 0.5
    assert metrics["DPRRetriever"]["map"] == 0.5
    assert metrics["DPRRetriever"]["recall_multi_hit"] == 0.5
    assert metrics["DPRRetriever"]["recall_single_hit"] == 0.5
    assert metrics["DPRRetriever"]["precision"] == 0.1
    assert metrics["DPRRetriever"]["ndcg"] == 0.5

    assert metrics["ESRetriever"]["mrr"] == 1.0
    assert metrics["ESRetriever"]["map"] == 1.0
    assert metrics["ESRetriever"]["recall_multi_hit"] == 1.0
    assert metrics["ESRetriever"]["recall_single_hit"] == 1.0
    assert metrics["ESRetriever"]["precision"] == 0.2
    assert metrics["ESRetriever"]["ndcg"] == 1.0

    assert metrics["QAReader"]["exact_match"] == 1.0
    assert metrics["QAReader"]["f1"] == 1.0


@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
@pytest.mark.parametrize("document_store_with_docs", ["memory"], indirect=True)
@pytest.mark.parametrize("reader", ["farm", "transformers"], indirect=True)
def test_empty_documents_dont_fail_pipeline(reader, retriever_with_docs, eval_labels):
    multilabels = eval_labels[:2]
    multilabels[0].labels[0].document.content = ""
    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever_with_docs)
    eval_result_integrated: EvaluationResult = pipeline.eval(labels=multilabels, add_isolated_node_eval=False)
    assert eval_result_integrated["Reader"]["answer"].iloc[0] == "Carla"
    eval_result_iso: EvaluationResult = pipeline.eval(labels=multilabels, add_isolated_node_eval=True)
    assert eval_result_iso["Reader"].loc[eval_result_iso["Reader"]["eval_mode"] == "isolated"]["answer"].iloc[0] == ""

    eval_batch_result_integrated: EvaluationResult = pipeline.eval_batch(
        labels=multilabels, add_isolated_node_eval=False
    )
    assert eval_batch_result_integrated["Reader"]["answer"].iloc[0] == "Carla"
    eval_batch_result_iso: EvaluationResult = pipeline.eval_batch(labels=multilabels, add_isolated_node_eval=True)
    assert (
        eval_batch_result_iso["Reader"]
        .loc[eval_batch_result_iso["Reader"]["eval_mode"] == "isolated"]["answer"]
        .iloc[0]
        == ""
    )


@pytest.mark.unit
def test_load_legacy_evaluation_result(tmp_path):
    legacy_csv = Path(tmp_path) / "legacy.csv"
    with open(legacy_csv, "w") as legacy_csv:
        columns = ["answer", "document_id", "custom_document_id", "gold_document_contents", "content"]
        writer = DictWriter(legacy_csv, fieldnames=columns)
        writer.writeheader()
        writer.writerow(
            {
                "answer": "answer",
                "document_id": Document("test").id,
                "custom_document_id": "custom_id",
                "gold_document_contents": ["gold", "document", "contents"],
                "content": "content",
            }
        )

    eval_result = EvaluationResult.load(tmp_path)
    assert "legacy" in eval_result
    assert len(eval_result["legacy"]) == 1
    assert eval_result["legacy"]["answer"].iloc[0] == "answer"
    assert eval_result["legacy"]["document_ids"].iloc[0] == [Document("test").id]
    assert eval_result["legacy"]["custom_document_ids"].iloc[0] == ["custom_id"]
    assert eval_result["legacy"]["gold_contexts"].iloc[0] == ["gold", "document", "contents"]
    assert eval_result["legacy"]["context"].iloc[0] == "content"

    assert "document_id" not in eval_result["legacy"]
    assert "custom_document_id" not in eval_result["legacy"]
    assert "gold_document_contents" not in eval_result["legacy"]
    assert "content" not in eval_result["legacy"]


@pytest.mark.unit
def test_load_evaluation_result(tmp_path):
    eval_result_csv = Path(tmp_path) / "Reader.csv"
    with open(eval_result_csv, "w") as eval_result_csv:
        columns = [
            "multilabel_id",
            "query",
            "filters",
            "gold_answers",
            "answer",
            "context",
            "exact_match",
            "f1",
            "exact_match_context_scope",
            "f1_context_scope",
            "exact_match_document_id_scope",
            "f1_document_id_scope",
            "exact_match_document_id_and_context_scope",
            "f1_document_id_and_context_scope",
            "gold_contexts",
            "rank",
            "document_ids",
            "gold_document_ids",
            "offsets_in_document",
            "gold_offsets_in_documents",
            "offsets_in_context",
            "gold_offsets_in_contexts",
            "gold_answers_exact_match",
            "gold_answers_f1",
            "gold_documents_id_match",
            "gold_contexts_similarity",
            "type",
            "node",
            "eval_mode",
            "index",
        ]
        writer = DictWriter(eval_result_csv, fieldnames=columns)
        writer.writeheader()
        writer.writerow(
            {
                "multilabel_id": "ddc1562602f2d6d895b91e53f83e4c16",
                "query": "who is written in the book of life",
                "filters": "b'null'",
                "gold_answers": "['every person who is destined for Heaven or the World to Come', 'all people considered righteous before God']",
                "answer": None,
                "context": None,
                "exact_match": 0.0,
                "f1": 0.0,
                "exact_match_context_scope": 0.0,
                "f1_context_scope": 0.0,
                "exact_match_document_id_scope": 0.0,
                "f1_document_id_scope": 0.0,
                "exact_match_document_id_and_context_scope": 0.0,
                "f1_document_id_and_context_scope": 0.0,
                "gold_contexts": "['Book of Life - wikipedia Book of Life Jump to: navigation, search...']",
                "rank": 1.0,
                "document_ids": None,
                "gold_document_ids": "['de2fd2f109e11213af1ea189fd1488a3-0', 'de2fd2f109e11213af1ea189fd1488a3-0']",
                "offsets_in_document": "[{'start': 0, 'end': 0}]",
                "gold_offsets_in_documents": "[{'start': 374, 'end': 434}, {'start': 1107, 'end': 1149}]",
                "offsets_in_context": "[{'start': 0, 'end': 0}]",
                "gold_offsets_in_contexts": "[{'start': 374, 'end': 434}, {'start': 1107, 'end': 1149}]",
                "gold_answers_exact_match": "[0, 0]",
                "gold_answers_f1": "[0, 0]",
                "gold_documents_id_match": "[0.0, 0.0]",
                "gold_contexts_similarity": "[0.0, 0.0]",
                "type": "answer",
                "node": "Reader",
                "eval_mode": "integrated",
            }
        )

    eval_result = EvaluationResult.load(tmp_path)
    known_result = {
        "multilabel_id": {0: "ddc1562602f2d6d895b91e53f83e4c16"},
        "query": {0: "who is written in the book of life"},
        "filters": {0: b"null"},
        "gold_answers": {
            0: [
                "every person who is destined for Heaven or the World to Come",
                "all people considered righteous before God",
            ]
        },
        "answer": {0: None},
        "context": {0: None},
        "exact_match": {0: 0.0},
        "f1": {0: 0.0},
        "exact_match_context_scope": {0: 0.0},
        "f1_context_scope": {0: 0.0},
        "exact_match_document_id_scope": {0: 0.0},
        "f1_document_id_scope": {0: 0.0},
        "exact_match_document_id_and_context_scope": {0: 0.0},
        "f1_document_id_and_context_scope": {0: 0.0},
        "gold_contexts": {0: ["Book of Life - wikipedia Book of Life Jump to: navigation, search..."]},
        "rank": {0: 1.0},
        "document_ids": {0: None},
        "gold_document_ids": {0: ["de2fd2f109e11213af1ea189fd1488a3-0", "de2fd2f109e11213af1ea189fd1488a3-0"]},
        "offsets_in_document": {0: [{"start": 0, "end": 0}]},
        "gold_offsets_in_documents": {0: [{"start": 374, "end": 434}, {"start": 1107, "end": 1149}]},
        "offsets_in_context": {0: [{"start": 0, "end": 0}]},
        "gold_offsets_in_contexts": {0: [{"start": 374, "end": 434}, {"start": 1107, "end": 1149}]},
        "gold_answers_exact_match": {0: [0, 0]},
        "gold_answers_f1": {0: [0, 0]},
        "gold_documents_id_match": {0: [0.0, 0.0]},
        "gold_contexts_similarity": {0: [0.0, 0.0]},
        "type": {0: "answer"},
        "node": {0: "Reader"},
        "eval_mode": {0: "integrated"},
        "index": {0: None},
    }
    assert "Reader" in eval_result
    assert len(eval_result) == 1
    assert eval_result["Reader"].to_dict() == known_result
