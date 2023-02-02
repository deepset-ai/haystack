import logging
import pytest
import sys
from haystack.document_stores.memory import InMemoryDocumentStore
from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack.nodes.preprocessor import PreProcessor
from haystack.nodes.evaluator import EvalAnswers, EvalDocuments
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
from haystack.schema import Answer, Document, EvaluationResult, Label, MultiLabel, Span

from ..conftest import SAMPLES_PATH


@pytest.mark.skipif(sys.platform in ["win32", "cygwin"], reason="Causes OOM on windows github runner")
@pytest.mark.parametrize("document_store_with_docs", ["memory"], indirect=True)
@pytest.mark.parametrize("retriever_with_docs", ["embedding"], indirect=True)
def test_generativeqa_calculate_metrics(
    document_store_with_docs: InMemoryDocumentStore, rag_generator, retriever_with_docs
):
    document_store_with_docs.update_embeddings(retriever=retriever_with_docs)
    pipeline = GenerativeQAPipeline(generator=rag_generator, retriever=retriever_with_docs)
    eval_result: EvaluationResult = pipeline.eval(labels=EVAL_LABELS, params={"Retriever": {"top_k": 5}})

    metrics = eval_result.calculate_metrics(document_scope="document_id")

    assert "Retriever" in eval_result
    assert "Generator" in eval_result
    assert len(eval_result) == 2

    assert metrics["Retriever"]["mrr"] == 0.5
    assert metrics["Retriever"]["map"] == 0.5
    assert metrics["Retriever"]["recall_multi_hit"] == 0.5
    assert metrics["Retriever"]["recall_single_hit"] == 0.5
    assert metrics["Retriever"]["precision"] == 0.1
    assert metrics["Retriever"]["ndcg"] == 0.5
    assert metrics["Generator"]["exact_match"] == 0.0
    assert metrics["Generator"]["f1"] == 1.0 / 3


@pytest.mark.skipif(sys.platform in ["win32", "cygwin"], reason="Causes OOM on windows github runner")
@pytest.mark.parametrize("document_store_with_docs", ["memory"], indirect=True)
@pytest.mark.parametrize("retriever_with_docs", ["embedding"], indirect=True)
def test_summarizer_calculate_metrics(document_store_with_docs: ElasticsearchDocumentStore, retriever_with_docs):
    document_store_with_docs.update_embeddings(retriever=retriever_with_docs)
    summarizer = TransformersSummarizer(model_name_or_path="sshleifer/distill-pegasus-xsum-16-4", use_gpu=False)
    pipeline = SearchSummarizationPipeline(
        retriever=retriever_with_docs, summarizer=summarizer, return_in_answer_format=True
    )
    eval_result: EvaluationResult = pipeline.eval(
        labels=EVAL_LABELS, params={"Retriever": {"top_k": 5}}, context_matching_min_length=10
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


@pytest.mark.parametrize("document_store", ["elasticsearch", "faiss", "memory", "milvus"], indirect=True)
@pytest.mark.parametrize("batch_size", [None, 20])
def test_add_eval_data(document_store, batch_size):
    # add eval data (SQUAD format)
    document_store.add_eval_data(
        filename=SAMPLES_PATH / "squad" / "small.json",
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
    assert label.answer.document_id == label.document.id

    # check combination
    doc = document_store.get_document_by_id(label.document.id)
    start = label.answer.offsets_in_document[0].start
    end = label.answer.offsets_in_document[0].end
    assert end == start + len(label.answer.answer)
    assert doc.content[start:end] == "France"


@pytest.mark.parametrize("document_store", ["elasticsearch", "faiss", "memory", "milvus"], indirect=True)
@pytest.mark.parametrize("reader", ["farm"], indirect=True)
@pytest.mark.parametrize("use_confidence_scores", [True, False])
def test_eval_reader(reader, document_store, use_confidence_scores):
    # add eval data (SQUAD format)
    document_store.add_eval_data(
        filename=SAMPLES_PATH / "squad" / "tiny.json",
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


@pytest.mark.elasticsearch
@pytest.mark.parametrize("document_store", ["elasticsearch"], indirect=True)
@pytest.mark.parametrize("open_domain", [True, False])
@pytest.mark.parametrize("retriever", ["bm25"], indirect=True)
def test_eval_elastic_retriever(document_store, open_domain, retriever):
    # add eval data (SQUAD format)
    document_store.add_eval_data(
        filename=SAMPLES_PATH / "squad" / "tiny.json",
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


# TODO simplify with a mock retriever and make it independent of elasticsearch documentstore
@pytest.mark.elasticsearch
@pytest.mark.parametrize("document_store", ["elasticsearch"], indirect=True)
@pytest.mark.parametrize("reader", ["farm"], indirect=True)
@pytest.mark.parametrize("retriever", ["bm25"], indirect=True)
def test_eval_pipeline(document_store, reader, retriever):
    # add eval data (SQUAD format)
    document_store.add_eval_data(
        filename=SAMPLES_PATH / "squad" / "tiny.json",
        doc_index=document_store.index,
        label_index=document_store.label_index,
    )

    labels = document_store.get_all_labels_aggregated(drop_negative_labels=True, drop_no_answers=False)

    eval_retriever = EvalDocuments()
    eval_reader = EvalAnswers(sas_model="sentence-transformers/paraphrase-MiniLM-L3-v2", debug=True)
    eval_reader_cross = EvalAnswers(sas_model="cross-encoder/stsb-TinyBERT-L-4", debug=True)
    eval_reader_vanila = EvalAnswers()

    assert document_store.get_document_count() == 2
    p = Pipeline()
    p.add_node(component=retriever, name="ESRetriever", inputs=["Query"])
    p.add_node(component=eval_retriever, name="EvalDocuments", inputs=["ESRetriever"])
    p.add_node(component=reader, name="QAReader", inputs=["EvalDocuments"])
    p.add_node(component=eval_reader, name="EvalAnswers", inputs=["QAReader"])
    p.add_node(component=eval_reader_cross, name="EvalAnswers_cross", inputs=["QAReader"])
    p.add_node(component=eval_reader_vanila, name="EvalAnswers_vanilla", inputs=["QAReader"])
    for l in labels:
        res = p.run(query=l.query, labels=l)
    assert eval_retriever.recall == 1.0
    assert eval_reader.top_k_f1 == pytest.approx(0.75)
    assert eval_reader.top_k_em == 0.5
    assert eval_reader.top_k_sas == pytest.approx(0.87586, 1e-4)
    assert eval_reader_cross.top_k_sas == pytest.approx(0.71063, 1e-4)
    assert eval_reader.top_k_em == eval_reader_vanila.top_k_em


@pytest.mark.parametrize("document_store", ["elasticsearch", "faiss", "memory", "milvus"], indirect=True)
def test_eval_data_split_word(document_store):
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
        filename=SAMPLES_PATH / "squad" / "tiny.json",
        doc_index=document_store.index,
        label_index=document_store.label_index,
        preprocessor=preprocessor,
    )
    labels = document_store.get_all_labels_aggregated()
    docs = document_store.get_all_documents()
    assert len(docs) == 5
    assert len(set(labels[0].document_ids)) == 2


@pytest.mark.parametrize("document_store", ["elasticsearch", "faiss", "memory", "milvus"], indirect=True)
def test_eval_data_split_passage(document_store):
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
        filename=SAMPLES_PATH / "squad" / "tiny_passages.json",
        doc_index=document_store.index,
        label_index=document_store.label_index,
        preprocessor=preprocessor,
    )
    docs = document_store.get_all_documents()
    assert len(docs) == 2
    assert len(docs[1].content) == 56


EVAL_LABELS = [
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
                answer=Answer(answer="Carla", offsets_in_context=[Span(11, 16)]),
                document=Document(
                    id="something_else", content_type="text", content="My name is Carla and I live in Munich"
                ),
                is_correct_answer=True,
                is_correct_document=True,
                origin="gold-label",
            )
        ]
    ),
]

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


@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
@pytest.mark.parametrize("document_store_with_docs", ["memory"], indirect=True)
@pytest.mark.parametrize("reader", ["farm"], indirect=True)
def test_extractive_qa_eval(reader, retriever_with_docs, tmp_path):
    labels = EVAL_LABELS[:1]

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
        "gold_answers_match",  # doc-specific
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
        "document_id",  # generic
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
@pytest.mark.parametrize("reader", ["farm"], indirect=True)
def test_extractive_qa_eval_multiple_queries(reader, retriever_with_docs, tmp_path):
    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever_with_docs)
    eval_result: EvaluationResult = pipeline.eval(labels=EVAL_LABELS, params={"Retriever": {"top_k": 5}})

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
def test_extractive_qa_eval_sas(reader, retriever_with_docs):
    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever_with_docs)
    eval_result: EvaluationResult = pipeline.eval(
        labels=EVAL_LABELS,
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
def test_reader_eval_in_pipeline(reader):
    pipeline = Pipeline()
    pipeline.add_node(component=reader, name="Reader", inputs=["Query"])
    eval_result: EvaluationResult = pipeline.eval(
        labels=EVAL_LABELS,
        documents=[[label.document for label in multilabel.labels] for multilabel in EVAL_LABELS],
        params={},
    )

    metrics = eval_result.calculate_metrics(document_scope="document_id")

    assert metrics["Reader"]["exact_match"] == 1.0
    assert metrics["Reader"]["f1"] == 1.0


@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
@pytest.mark.parametrize("document_store_with_docs", ["memory"], indirect=True)
def test_extractive_qa_eval_document_scope(retriever_with_docs):
    pipeline = DocumentSearchPipeline(retriever=retriever_with_docs)
    eval_result: EvaluationResult = pipeline.eval(
        labels=EVAL_LABELS,
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
def test_extractive_qa_eval_answer_scope(reader, retriever_with_docs):
    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever_with_docs)
    eval_result: EvaluationResult = pipeline.eval(
        labels=EVAL_LABELS,
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
def test_extractive_qa_eval_answer_document_scope_combinations(reader, retriever_with_docs, caplog):
    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever_with_docs)
    eval_result: EvaluationResult = pipeline.eval(
        labels=EVAL_LABELS,
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
def test_extractive_qa_eval_simulated_top_k_reader(reader, retriever_with_docs):
    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever_with_docs)
    eval_result: EvaluationResult = pipeline.eval(
        labels=EVAL_LABELS,
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
def test_extractive_qa_eval_simulated_top_k_retriever(reader, retriever_with_docs):
    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever_with_docs)
    eval_result: EvaluationResult = pipeline.eval(labels=EVAL_LABELS, params={"Retriever": {"top_k": 5}})

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
def test_extractive_qa_eval_simulated_top_k_reader_and_retriever(reader, retriever_with_docs):
    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever_with_docs)
    eval_result: EvaluationResult = pipeline.eval(labels=EVAL_LABELS, params={"Retriever": {"top_k": 10}})

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
def test_extractive_qa_eval_isolated(reader, retriever_with_docs):
    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever_with_docs)
    eval_result: EvaluationResult = pipeline.eval(
        labels=EVAL_LABELS,
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
def test_document_search_calculate_metrics(retriever_with_docs):
    pipeline = DocumentSearchPipeline(retriever=retriever_with_docs)
    eval_result: EvaluationResult = pipeline.eval(labels=EVAL_LABELS, params={"Retriever": {"top_k": 5}})

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
def test_document_search_isolated(retriever_with_docs):
    pipeline = DocumentSearchPipeline(retriever=retriever_with_docs)
    # eval run must not fail even though no node supports add_isolated_node_eval
    eval_result: EvaluationResult = pipeline.eval(
        labels=EVAL_LABELS, params={"Retriever": {"top_k": 5}}, add_isolated_node_eval=True
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
def test_faq_calculate_metrics(retriever_with_docs):
    pipeline = FAQPipeline(retriever=retriever_with_docs)
    eval_result: EvaluationResult = pipeline.eval(labels=EVAL_LABELS, params={"Retriever": {"top_k": 5}})

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
def test_extractive_qa_eval_translation(reader, retriever_with_docs):

    # FIXME it makes no sense to have DE->EN input and DE->EN output, right?
    #  Yet switching direction breaks the test. TO BE FIXED.
    input_translator = TransformersTranslator(model_name_or_path="Helsinki-NLP/opus-mt-de-en")
    output_translator = TransformersTranslator(model_name_or_path="Helsinki-NLP/opus-mt-de-en")

    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever_with_docs)
    pipeline = TranslationWrapperPipeline(
        input_translator=input_translator, output_translator=output_translator, pipeline=pipeline
    )
    eval_result: EvaluationResult = pipeline.eval(labels=EVAL_LABELS, params={"Retriever": {"top_k": 5}})

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
def test_question_generation_eval(retriever_with_docs, question_generator):
    pipeline = RetrieverQuestionGenerationPipeline(retriever=retriever_with_docs, question_generator=question_generator)

    eval_result: EvaluationResult = pipeline.eval(labels=EVAL_LABELS, params={"Retriever": {"top_k": 5}})

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
def test_qa_multi_retriever_pipeline_eval(document_store_with_docs, reader):
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
    labels = EVAL_LABELS + [
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
def test_multi_retriever_pipeline_eval(document_store_with_docs):
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
    labels = EVAL_LABELS + [
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
def test_multi_retriever_pipeline_with_asymmetric_qa_eval(document_store_with_docs, reader):
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
    labels = EVAL_LABELS + [
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
def test_empty_documents_dont_fail_pipeline(reader, retriever_with_docs):
    multilabels = EVAL_LABELS[:2]
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
